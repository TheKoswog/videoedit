import tkinter as tk
from tkinter import filedialog, messagebox
import librosa
import numpy as np
from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    AudioFileClip,
    CompositeAudioClip,
)
from scipy.signal import find_peaks
import random
import threading

def process_files(audio_path, video_path, output_path, progress_callback):
    try:
        # Load the song and get the sampling rate
        y, sr = librosa.load(audio_path)

        # Get the duration of the song (in seconds)
        song_duration = librosa.get_duration(y=y, sr=sr)

        # Analyze the bass frequencies of the song
        progress_callback("Calculating STFT...")
        S = np.abs(librosa.stft(y))

        # Get frequencies
        frequencies = librosa.fft_frequencies(sr=sr)

        # Select bass frequencies (between 20Hz and 250Hz)
        bass_freq_idx = np.where((frequencies >= 20) & (frequencies <= 250))[0]
        bass_spectrum = S[bass_freq_idx, :]

        # Calculate energy levels of the bass frequencies
        bass_energy = np.sum(bass_spectrum, axis=0)

        # Create the time axis
        times = librosa.frames_to_time(np.arange(len(bass_energy)), sr=sr)

        # Detect the peak points of bass energy levels
        progress_callback("Detecting peak points...")
        peaks, _ = find_peaks(bass_energy, height=np.max(bass_energy) * 0.7)

        # Get the times of the peak points
        peak_times = times[peaks]

        # Determine cut times (0, peak times, and the end of the song)
        cut_times = np.sort(np.concatenate(([0], peak_times, [song_duration])))

        # Load the video
        progress_callback("Loading video...")
        video = VideoFileClip(video_path)

        # Get the total duration of the video
        video_duration = video.duration

        # Create clips
        clips = []
        total_clips = len(cut_times) - 1
        for i in range(len(cut_times) - 1):
            start_time = cut_times[i]
            end_time = cut_times[i + 1]
            clip_duration = end_time - start_time

            if clip_duration <= 0:
                continue

            # Get a random time from the video for the desired duration
            random_start = random.uniform(0, max(0, video_duration - clip_duration))
            random_end = random_start + clip_duration

            # Create the clip
            clip = video.subclip(random_start, random_end)
            clips.append(clip)

            progress_callback(f"{i + 1}/{total_clips} clips processed...")

        # Concatenate the clips
        progress_callback("Concatenating clips...")
        final_video = concatenate_videoclips(clips, method='compose')

        # Get the original audio of the video
        original_audio = final_video.audio

        # Get the audio of the song
        progress_callback("Merging audio...")
        song_audio = AudioFileClip(audio_path).subclip(0, final_video.duration)

        # Merge the audios (you can adjust the volumes)
        mixed_audio = CompositeAudioClip([
            original_audio.volumex(0.5),
            song_audio.volumex(0.5)
        ])

        # Add the merged audio to the video
        final_video = final_video.set_audio(mixed_audio)

        # Save the final video
        progress_callback("Saving video...")
        final_video.write_videofile(
            output_path,
            fps=video.fps,
            audio_codec='aac',
            threads=4
        )

        progress_callback("Process completed!")
        messagebox.showinfo("Success", "The video was successfully created!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def browse_file(entry, file_type):
    if file_type == "audio":
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("MP3 Files", "*.mp3"), ("WAV Files", "*.wav"), ("All Files", "*.*")]
        )
    elif file_type == "video":
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("MP4 Files", "*.mp4"), ("AVI Files", "*.avi"), ("All Files", "*.*")]
        )
    else:
        file_path = filedialog.asksaveasfilename(
            title="Save Output File",
            defaultextension=".mp4",
            filetypes=[("MP4 Files", "*.mp4")]
        )
    entry.delete(0, tk.END)
    entry.insert(0, file_path)

def start_processing(audio_entry, video_entry, output_entry, progress_label):
    audio_path = audio_entry.get()
    video_path = video_entry.get()
    output_path = output_entry.get()

    if not audio_path or not video_path or not output_path:
        messagebox.showwarning("Missing Information", "Please select all file paths.")
        return

    # Use threading to prevent GUI from freezing during processing
    def run():
        process_files(audio_path, video_path, output_path, update_progress)

    def update_progress(message):
        progress_label.config(text=message)
        progress_label.update_idletasks()

    threading.Thread(target=run).start()

def create_gui():
    root = tk.Tk()
    root.title("Video and Audio Processor")

    # Select Audio File
    tk.Label(root, text="Audio File:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
    audio_entry = tk.Entry(root, width=50)
    audio_entry.grid(row=0, column=1, padx=10, pady=10)
    tk.Button(root, text="Browse", command=lambda: browse_file(audio_entry, "audio")).grid(row=0, column=2, padx=10, pady=10)

    # Select Video File
    tk.Label(root, text="Video File:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
    video_entry = tk.Entry(root, width=50)
    video_entry.grid(row=1, column=1, padx=10, pady=10)
    tk.Button(root, text="Browse", command=lambda: browse_file(video_entry, "video")).grid(row=1, column=2, padx=10, pady=10)

    # Select Output File
    tk.Label(root, text="Output File:").grid(row=2, column=0, padx=10, pady=10, sticky='e')
    output_entry = tk.Entry(root, width=50)
    output_entry.grid(row=2, column=1, padx=10, pady=10)
    tk.Button(root, text="Save", command=lambda: browse_file(output_entry, "output")).grid(row=2, column=2, padx=10, pady=10)

    # Start Processing Button
    tk.Button(root, text="Start Processing", command=lambda: start_processing(audio_entry, video_entry, output_entry, progress_label)).grid(row=3, column=1, pady=20)

    # Progress Label
    progress_label = tk.Label(root, text="Waiting...", fg="blue")
    progress_label.grid(row=4, column=0, columnspan=3, pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
