import pygame

def play_audio(file_path):
    # Initialize Pygame mixer
    pygame.mixer.init()

    # Load the audio file
    pygame.mixer.music.load(file_path)

    # Play the audio file
    pygame.mixer.music.play()

    # Wait until the audio is finished playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Example usage
audio_file_path = "sound/1000.wav"  # Replace this with the path to your audio file
play_audio(audio_file_path)
