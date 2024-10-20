import numpy as np
from tkinter import Tk, filedialog
from PIL import Image
import soundfile as sf

# Constants
SAMPLE_RATE = 48000  # Sample rate in Hz

# Base SSTV Mode Class
class SSTVMode:
    def __init__(self, name, num_lines, sync_freq, sync_duration, porch_freq, porch_duration,
                 color_freq_min, color_freq_max):
        self.name = name
        self.num_lines = num_lines
        self.sync_freq = sync_freq
        self.sync_duration = sync_duration
        self.porch_freq = porch_freq
        self.porch_duration = porch_duration
        self.color_freq_min = color_freq_min
        self.color_freq_max = color_freq_max

    def generate_signal(self, image_data):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def process_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        # Resize image to match the number of lines expected by the SSTV mode
        width, height = image.size
        aspect_ratio = width / height
        new_height = self.num_lines
        new_width = int(aspect_ratio * new_height)
        # Use Image.LANCZOS or Image.Resampling.LANCZOS depending on Pillow version
        try:
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        except AttributeError:
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        np_image = np.array(resized_image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        return np_image

    def generate_tone(self, frequency, duration):
        num_samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        return np.sin(2 * np.pi * frequency * t)

    def generate_line_signal(self, line_data, duration):
        num_samples = int(SAMPLE_RATE * duration)
        t_line = np.linspace(0, duration, num_samples, endpoint=False)
        frequencies = self.color_freq_min + line_data * (self.color_freq_max - self.color_freq_min)
        freq_time_array = np.interp(t_line, np.linspace(0, duration, len(line_data)), frequencies)
        phase = 2 * np.pi * np.cumsum(freq_time_array) / SAMPLE_RATE
        return np.sin(phase)

# Martin M2 Mode Class
class MartinM2Mode(SSTVMode):
    def __init__(self):
        super().__init__(
            name="Martin M2",
            num_lines=256,
            sync_freq=1200,
            sync_duration=0.004862,     # 4.862 ms
            porch_freq=1500,
            porch_duration=0.000572,    # 0.572 ms
            color_freq_min=1500,
            color_freq_max=2300,
        )
        self.color_channels = ["green", "blue", "red"]
        self.color_scan_duration = 0.073216  # 73.216 ms per color channel

    def generate_signal(self, image_data):
        signal_list = []
        num_lines = image_data.shape[0]

        for i in range(num_lines):
            # Sync Pulse
            signal_list.append(self.generate_tone(self.sync_freq, self.sync_duration))

            for channel_name in self.color_channels:
                # Porch
                signal_list.append(self.generate_tone(self.porch_freq, self.porch_duration))

                # Color Scan
                channel_index = {'red': 0, 'green': 1, 'blue': 2}[channel_name]
                color_line = image_data[i, :, channel_index]
                line_signal = self.generate_line_signal(color_line, self.color_scan_duration)
                signal_list.append(line_signal)

            # Progress Indicator
            progress_percentage = (i + 1) / num_lines * 100
            print(f"\rEncoding progress: {progress_percentage:.2f}%", end="")

        print("\nEncoding complete.")
        return np.concatenate(signal_list)

# Robot36 Mode Class
class Robot36Mode(SSTVMode):
    def __init__(self):
        super().__init__(
            name="Robot36",
            num_lines=240,
            sync_freq=1200,
            sync_duration=0.009,        # 9 ms
            porch_freq=1500,
            porch_duration=0.0015,      # 1.5 ms
            color_freq_min=1500,
            color_freq_max=2300,
        )
        self.vis_code = 0x08
        self.line_duration = 0.088       # 88 ms for Y scan
        self.chroma_duration = 0.044     # 44 ms for Cb and Cr scans

    def generate_vis_code_signal(self):
        # VIS code specifications
        bits = [0] + [int(b) for b in f"{self.vis_code:07b}"[::-1]]  # Start bit + 7 bits (LSB first)
        parity = sum(bits[1:]) % 2
        bits.append(parity)  # Parity bit
        bits.append(1)       # Stop bit

        signal = []
        bit_duration = 0.030  # 30 ms per bit
        for bit in bits:
            freq = 1100 if bit else 1300
            signal.append(self.generate_tone(freq, bit_duration))
        return np.concatenate(signal)

    def rgb_to_ycbcr(self, image_rgb):
        R = image_rgb[:, :, 0]
        G = image_rgb[:, :, 1]
        B = image_rgb[:, :, 2]

        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 0.5
        Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 0.5

        YCbCr = np.stack((Y, Cb, Cr), axis=-1)
        return YCbCr

    def generate_signal(self, image_data):
        signal_list = []

        # Generate VIS Code
        vis_signal = self.generate_vis_code_signal()
        signal_list.append(vis_signal)

        # Convert Image to YCbCr
        image_ycbcr = self.rgb_to_ycbcr(image_data)
        Y_channel = image_ycbcr[:, :, 0]
        Cb_channel = image_ycbcr[:, :, 1]
        Cr_channel = image_ycbcr[:, :, 2]

        num_lines = Y_channel.shape[0]

        # Process Fields
        for field in range(2):  # Field 0 and 1
            field_lines = range(field, num_lines, 2)
            num_field_lines = len(field_lines)

            # Process Y lines
            for idx, i in enumerate(field_lines):
                # Sync and Porch
                signal_list.append(self.generate_tone(self.sync_freq, self.sync_duration))
                signal_list.append(self.generate_tone(self.porch_freq, self.porch_duration))

                # Y Line
                line_signal = self.generate_line_signal(Y_channel[i, :], self.line_duration)
                signal_list.append(line_signal)

                # Progress Indicator
                total_lines = num_lines
                progress = (idx + 1 + field * num_field_lines) / total_lines * 100
                print(f"\rEncoding progress: {progress:.2f}%", end="")

            # Process Cb and Cr lines after Y lines
            for chroma_channel, chroma_data in [('Cb', Cb_channel), ('Cr', Cr_channel)]:
                # Sync and Porch
                signal_list.append(self.generate_tone(self.sync_freq, self.sync_duration))
                signal_list.append(self.generate_tone(self.porch_freq, self.porch_duration))

                # Average chroma over field lines
                chroma_line = chroma_data[field_lines, :].mean(axis=0)
                chroma_signal = self.generate_line_signal(chroma_line, self.chroma_duration)
                signal_list.append(chroma_signal)

        print("\nEncoding complete.")
        return np.concatenate(signal_list)

# Utility Functions
def get_file_path(dialog_function, *args, **kwargs):
    try:
        root = Tk()
        root.withdraw()
        root.lift()
        root.attributes('-topmost', True)
        file_path = dialog_function(*args, **kwargs)
        return file_path
    finally:
        root.destroy()

def select_image():
    return get_file_path(filedialog.askopenfilename, filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

def select_save_file():
    return get_file_path(filedialog.asksaveasfilename, defaultextension=".wav",
                         filetypes=[("WAV Files", "*.wav")], title="Save SSTV Signal As")

def select_sstv_mode():
    modes = {
        1: MartinM2Mode(),
        2: Robot36Mode(),
    }
    print("Available SSTV Modes:")
    for key, mode in modes.items():
        print(f"{key}. {mode.name}")
    while True:
        try:
            choice = int(input("Select the SSTV mode: "))
            if choice in modes:
                return modes[choice]
            else:
                print("Invalid choice. Please select a valid mode number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def save_sstv_signal(signal, file_name):
    # Normalize signal to [-1, 1]
    max_abs_value = np.max(np.abs(signal))
    if max_abs_value > 0:
        signal = signal / max_abs_value
    # Save the audio file
    sf.write(file_name, signal, SAMPLE_RATE)
    print(f"\nAudio saved to {file_name}")

# Main Function
def main():
    while True:
        choice = input("Choose an option:\n1. Encode Image to SSTV Audio\n2. Exit\nEnter your choice: ")
        if choice == '1':
            # Select SSTV mode
            mode = select_sstv_mode()
            # Select image to encode
            file_path = select_image()
            if not file_path:
                print("No image selected. Returning to menu...")
                continue
            # Process the image
            image_data = mode.process_image(file_path)
            # Generate SSTV signal
            sstv_signal = mode.generate_signal(image_data)
            # Select output file to save
            output_file = select_save_file()
            if not output_file:
                print("No output file selected. Returning to menu...")
                continue
            # Save SSTV signal to audio file
            save_sstv_signal(sstv_signal, output_file)
            print("Encoding completed and saved.")
        elif choice == '2':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
