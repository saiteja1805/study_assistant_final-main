import os
import openai
import gradio as gr
from pytube import YouTube
from moviepy.editor import VideoFileClip

# Load OpenAI API key from a file
with open('openai_Key.txt', 'r') as f:
    api_key = f.read().strip('\n')
    assert api_key.startswith('sk-'), "Please enter a valid OpenAI API key"
openai.api_key = api_key


def transcribe(audio_file, not_english=True):
    try:
        if not os.path.exists(audio_file):
            return "The following file does not exist: " + audio_file

        if not_english:
            with open(audio_file, 'rb') as f:
                transcript = openai.Audio.translate('whisper-1', f)

        else:
            with open(audio_file, 'rb') as f:
                transcript = openai.Audio.transcribe('whisper-1', f)

        name, extension = os.path.splitext(audio_file)
        transcript_filename = f'transcript-{name}.txt'
        with open(transcript_filename, 'w') as f:
            f.write(transcript['text'])

        return transcript_filename

    except Exception as e:
        return "An error occurred during transcription: " + str(e)

def summarize(transcript_filename):
    try:
        if not os.path.exists(transcript_filename):
            return "Please remove your generated local audio file and transcription file and try again"

        with open(transcript_filename) as f:
            transcript = f.read()

        system_prompt = "Act as Expert one who can summarize any topic"
        prompt = f'''Create a summary of the following text.
        Text{transcript}

        Add a title to the summary.
        Your summary should be informative and factual, covering the most important aspects of the topic.
        Use BULLET POINTS if possible'''

        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=2024,
            temperature=1
        )

        return response['choices'][0]['message']['content']

    except Exception as e:
        return "An error occurred during summarization: " + str(e)



def process_youtube_video(url):
    try:
        video = YouTube(url)
        audio = video.streams.filter(only_audio=True).first()
        out_file = audio.download('aud')
        base, ext = os.path.splitext(out_file)
        new_file = 'test.mp3'
        os.rename(out_file, new_file)
        return new_file
    except Exception as e:
        return "An error occurred while downloading the YouTube video: " + str(e)

def process_local_video(file_path):
    try:
        video = VideoFileClip(file_path)
        audio = video.audio
        audio.write_audiofile('test.mp3')
        return 'test.mp3'
    except Exception as e:
        return "An error occurred while processing the video file: " + str(e)

def run_process(video_source, url_or_path, is_english):
    if video_source == 'youtube':
        audio_file = process_youtube_video(url_or_path)
    elif video_source == 'file':
        audio_file = process_local_video(url_or_path)
    else:
        return "Invalid input."

    if audio_file:
        transcript_file = transcribe(audio_file, not_english=(not is_english))
        if transcript_file:
            summary = summarize(transcript_file)
            return summary

    return "Failed to process the audio or generate a summary."
