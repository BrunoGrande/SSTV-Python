import tkinter as tk
from tkinter import filedialog
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
import queue
import threading
from scipy import signal
from scipy.io import wavfile


class SSTVMode:
    PIXEL_FREQ_RANGE = (1500, 2300)  # Pixel frequency range for all modes

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def decode(self, audio_data):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def detect_sync_pulses(self, audio_data, sync_frequency=1200, threshold=0.7):
        """Detect sync pulses using the Goertzel algorithm."""
        sample_rate = self.sample_rate
        block_size = int(0.01 * sample_rate)  # 10 ms blocks
        sync_indices = []

        k = int(0.5 + (block_size * sync_frequency) / sample_rate)
        omega = (2 * np.pi * k) / block_size
        coeff = 2 * np.cos(omega)

        q1 = q2 = 0
        for i in range(0, len(audio_data) - block_size, block_size):
            q1 = q2 = 0
            for j in range(block_size):
                q0 = coeff * q1 - q2 + audio_data[i + j]
                q2 = q1
                q1 = q0
            magnitude = q1**2 + q2**2 - q1 * q2 * coeff
            normalized_magnitude = magnitude / block_size

            if normalized_magnitude > threshold:
                sync_indices.append(i)
        print(f"Total sync pulses detected: {len(sync_indices)}")
        return sync_indices

    def demodulate_frequency(self, audio_segment):
        """Demodulate frequency using zero-crossing detection."""
        zero_crossings = np.where(np.diff(np.sign(audio_segment)))[0]
        duration = len(audio_segment) / self.sample_rate
        frequency = len(zero_crossings) / (2 * duration)

        # Map the frequency to pixel intensity
        pixel_value = np.interp(frequency, [1500, 2300], [0, 255])
        return pixel_value

    def extract_line_segments(self, audio_data, sync_indices, line_duration):
        """Extract audio segments corresponding to each image line."""
        samples_per_line = int(line_duration * self.sample_rate)
        line_segments = []
        for index in sync_indices:
            line_segment = audio_data[index:index + samples_per_line]
            if len(line_segment) == samples_per_line:
                line_segments.append(line_segment)
        return line_segments


class Robot36(SSTVMode):
    """Robot36 SSTV mode class with improved decoding."""
    LINE_DURATION = 0.088  # 88 ms for Y scan
    CHROMA_DURATION = 0.044  # 44 ms for Cb/Cr scans

    def __init__(self, sample_rate):
        super().__init__(sample_rate)

    def decode(self, audio_data):
        """Decode SSTV image from Robot36 mode."""
        sync_indices = self.detect_sync_pulses(audio_data)
        if len(sync_indices) == 0:
            print("No sync pulses detected.")
            return None

        # Extract Y lines
        y_line_segments = self.extract_line_segments(
            audio_data, sync_indices, self.LINE_DURATION)

        # Extract Cb and Cr lines
        cb_cr_line_segments = []
        for index in sync_indices:
            cb_start = index + int(self.LINE_DURATION * self.sample_rate)
            cr_start = cb_start + int(self.CHROMA_DURATION * self.sample_rate)

            cb_segment = audio_data[cb_start:cb_start + int(self.CHROMA_DURATION * self.sample_rate)]
            cr_segment = audio_data[cr_start:cr_start + int(self.CHROMA_DURATION * self.sample_rate)]

            if len(cb_segment) == int(self.CHROMA_DURATION * self.sample_rate) and \
               len(cr_segment) == int(self.CHROMA_DURATION * self.sample_rate):
                cb_cr_line_segments.append((cb_segment, cr_segment))

        # Ensure we have matching Y, Cb, and Cr lines
        num_lines = min(len(y_line_segments), len(cb_cr_line_segments))
        image = np.zeros((num_lines, 320, 3), dtype=np.uint8)  # Adjust width as needed

        for i in range(num_lines):
            y_line = y_line_segments[i]
            cb_line, cr_line = cb_cr_line_segments[i]

            # Demodulate Y, Cb, and Cr
            y_pixels = self.demodulate_line(y_line)
            cb_pixels = self.demodulate_line(cb_line)
            cr_pixels = self.demodulate_line(cr_line)

            # Convert YCbCr to RGB
            image[i] = self.ycbcr_to_rgb(y_pixels, cb_pixels, cr_pixels)
            print(f"Decoded line {i+1}/{num_lines}")

        return image

    def demodulate_line(self, line_audio):
        """Demodulate a line to get pixel values."""
        segment_length = int(len(line_audio) / 320)  # Adjust pixel width
        pixels = []
        for i in range(320):
            segment = line_audio[i * segment_length:(i + 1) * segment_length]
            pixel_value = self.demodulate_frequency(segment)
            pixels.append(pixel_value)
        return np.array(pixels)

    def ycbcr_to_rgb(self, y_pixels, cb_pixels, cr_pixels):
        """Convert YCbCr pixels to RGB."""
        Y = y_pixels / 255.0
        Cb = cb_pixels / 255.0 - 0.5
        Cr = cr_pixels / 255.0 - 0.5

        R = Y + 1.402 * Cr
        G = Y - 0.344136 * Cb - 0.714136 * Cr
        B = Y + 1.772 * Cb

        rgb = np.stack((R, G, B), axis=-1)
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        return rgb


class MartinM2(SSTVMode):
    """MartinM2 SSTV mode class."""
    LINE_DURATION = 0.146432  # Line duration for Martin M2 (146.432 ms)

    def __init__(self, sample_rate):
        super().__init__(sample_rate)

    def decode(self, audio_data):
        """Decode SSTV image from Martin M2 mode."""
        samples_per_line = int(self.LINE_DURATION * self.sample_rate)
        image = cp.zeros((256, 320, 3))  # 320x256 RGB image

        sync_indices = self.detect_sync_pulses(audio_data, samples_per_line)

        # Decode RGB lines
        for line_idx, start_idx in enumerate(sync_indices[:256]):
            line_audio = audio_data[start_idx:start_idx + samples_per_line]
            image[line_idx] = self.decode_line(line_audio)

        return cp.asnumpy(image)

    def decode_line(self, line_audio):
        """Decode a single line into RGB pixel values."""
        segment_length = len(line_audio) // 3
        red_channel = self.demodulate_color_channel(line_audio[:segment_length], self.PIXEL_FREQ_RANGE[0], self.PIXEL_FREQ_RANGE[1])
        green_channel = self.demodulate_color_channel(line_audio[segment_length:2 * segment_length], self.PIXEL_FREQ_RANGE[0], self.PIXEL_FREQ_RANGE[1])
        blue_channel = self.demodulate_color_channel(line_audio[2 * segment_length:], self.PIXEL_FREQ_RANGE[0], self.PIXEL_FREQ_RANGE[1])
        return cp.stack([red_channel, green_channel, blue_channel], axis=-1)

    def demodulate_color_channel(self, audio_segment, start_freq, end_freq):
        """Convert the frequencies in the audio to pixel values."""
        # Convert the audio_segment to a cupy array
        audio_segment_cp = cp.asarray(audio_segment)

        # Perform FFT on the cupy array
        fft_data = cp.abs(cp.fft.rfft(audio_segment_cp))
        frequencies = cp.fft.rfftfreq(len(audio_segment_cp), d=1.0 / self.sample_rate)

        # Convert start_freq and end_freq to cupy arrays
        start_end_freq_cp = cp.asarray([start_freq, end_freq])
        pixel_range_cp = cp.asarray([0, 255])

        # Interpolate the frequencies to map them to pixel values
        pixel_values = cp.interp(frequencies, start_end_freq_cp, pixel_range_cp)

        return pixel_values[:320]  # Assuming 320 pixels per line




class SSTVDecoder:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.mode = None

    def set_mode(self, mode: SSTVMode):
        """Set the SSTV mode for decoding."""
        self.mode = mode

    def set_sample_rate(self, sample_rate):
        """Set the sample rate for decoding."""
        self.sample_rate = sample_rate

    def decode(self, audio_data):
        """Decode the SSTV image using the selected mode."""
        if self.mode is None:
            raise ValueError("No SSTV mode selected")
        return self.mode.decode(audio_data)


class SSTVDecoderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SSTV Decoder")
        self.root.geometry("800x600")
        self.current_mode = tk.StringVar(value="Robot36")
        self.sample_rate = 44100

        # SSTV Decoder instance
        self.decoder = SSTVDecoder(self.sample_rate)

        # Create radio buttons for mode selection
        self.radio_robot36 = tk.Radiobutton(root, text="Robot36", variable=self.current_mode, value="Robot36", command=self.update_mode)
        self.radio_robot36.pack()

        self.radio_martin_m2 = tk.Radiobutton(root, text="Martin M2", variable=self.current_mode, value="MartinM2", command=self.update_mode)
        self.radio_martin_m2.pack()

        # Create buttons for controls
        self.load_button = tk.Button(root, text="Load Audio File", command=self.load_audio_file)
        self.load_button.pack()

        self.mic_button = tk.Button(root, text="Use Microphone", command=self.use_microphone)
        self.mic_button.pack()

        # Buttons to change sample rate
        self.rate_44100_button = tk.Button(root, text="Set Sample Rate to 44100 Hz", command=self.set_sample_rate_44100)
        self.rate_44100_button.pack()

        self.rate_48000_button = tk.Button(root, text="Set Sample Rate to 48000 Hz", command=self.set_sample_rate_48000)
        self.rate_48000_button.pack()

        # Canvas for the SSTV image decoding
        self.image_figure = plt.Figure(figsize=(4, 4), dpi=100)
        self.image_canvas = FigureCanvasTkAgg(self.image_figure, root)
        self.image_canvas.get_tk_widget().pack()

        # Queue for audio data
        self.audio_queue = queue.Queue()

        # Thread for audio processing
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.start()

    def update_mode(self):
        """Update the decoding mode."""
        mode = self.current_mode.get()
        if mode == "Robot36":
            self.decoder.set_mode(Robot36(self.sample_rate))
        elif mode == "MartinM2":
            self.decoder.set_mode(MartinM2(self.sample_rate))

    def plot_sstv_image(self, image):
        """Update the SSTV image."""
        ax = self.image_figure.add_subplot(111)
        ax.clear()
        ax.imshow(image, aspect='auto')
        self.image_canvas.draw()

    def load_audio_file(self):
        """Open file dialog to load an audio file and process it."""
        audio_file = filedialog.askopenfilename(title="Open Audio File", filetypes=[("WAV Files", "*.wav")])
        if audio_file:
            rate, audio_data = wavfile.read(audio_file)
            self.sample_rate = rate
            self.decode_sstv_from_audio(audio_data)

    def use_microphone(self):
        """Capture real-time audio from the microphone and decode it."""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Overflow detected: {status}")
            audio_data = np.frombuffer(indata, dtype=np.float32)
            self.audio_queue.put(audio_data)

        stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=self.sample_rate, blocksize=4096)
        with stream:
            sd.sleep(10000)

    def process_audio(self):
        """Continuously process audio data from the queue."""
        while True:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                self.decode_sstv_from_audio(audio_data)

    def decode_sstv_from_audio(self, audio_data):
        """Decode SSTV image from audio data using selected mode."""
        self.update_mode()
        decoded_image = self.decoder.decode(audio_data)
        if decoded_image is not None:
            self.plot_sstv_image(decoded_image)

    def set_sample_rate_44100(self):
        """Set the sample rate to 44100 Hz."""
        self.sample_rate = 44100
        self.decoder.set_sample_rate(44100)
        print("Sample rate set to 44100 Hz")

    def set_sample_rate_48000(self):
        """Set the sample rate to 48000 Hz."""
        self.sample_rate = 48000
        self.decoder.set_sample_rate(48000)
        print("Sample rate set to 48000 Hz")


if __name__ == "__main__":
    root = tk.Tk()
    app = SSTVDecoderApp(root)
    root.mainloop()
