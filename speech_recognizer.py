import speech_recognition as sr
import os
import sys
from datetime import datetime

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    col = True
except:
    col = False

def pr(text, color="white", bold=False):
    if not col:
        print(text)
        return
    c = {
        "cyan": Fore.CYAN,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "red": Fore.RED,
        "blue": Fore.BLUE,
        "white": Fore.WHITE,
    }
    clr = c.get(color, Fore.WHITE)
    bld = Style.BRIGHT if bold else ""
    print(f"{bld}{clr}{text}{Style.RESET_ALL}")


# mic input function
def use_mic():
    r = sr.Recognizer()
    pr("Mic will start in a sec.. get ready", "yellow")
    input("Press enter to start recording: ")

    with sr.Microphone() as src:
        pr("adjusting for noise..", "blue")
        r.adjust_for_ambient_noise(src, duration=2)
        pr("GO! speak now", "green", bold=True)
        try:
            audio = r.listen(src, timeout=10, phrase_time_limit=15)
            pr("got audio, working on it..", "cyan")
        except sr.WaitTimeoutError:
            pr("didnt hear anything, try again", "red")
            return None, None
    return audio, r


# audio file function
def use_file(path):
    if not os.path.exists(path):
        pr("cant find that file bro", "red")
        return None, None

    r = sr.Recognizer()
    pr("loading the file..", "blue")
    with sr.AudioFile(path) as src:
        r.adjust_for_ambient_noise(src)
        audio = r.record(src)
    pr("done loading", "cyan")
    return audio, r


# convert speech to text
def get_text(audio, r):
    try:
        result = r.recognize_google(audio)
        return result
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        pr("api not working, check internet", "red")
        return None


def show_output(text, src):
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    words = len(text.split())

    print()
    pr("=" * 55, "cyan", bold=True)
    pr("     SPEECH RECOGNITION SYSTEM", "cyan", bold=True)
    pr("     Codetech Internship - Task 2", "cyan")
    pr("=" * 55, "cyan", bold=True)
    print()
    pr(f"  Source    : {src}", "blue")
    pr(f"  Time      : {now}", "blue")
    pr(f"  Words     : {words}", "blue")
    print()
    pr("TRANSCRIPTION OUTPUT:", "green", bold=True)
    pr("-" * 55, "green")
    print(text)
    print()
    pr("=" * 55, "cyan", bold=True)
    print()


def save_it(text, src):
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    with open("transcription_output.txt", "w", encoding="utf-8") as f:
        f.write("SPEECH RECOGNITION - CODETECH INTERNSHIP TASK 2\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Source : {src}\n")
        f.write(f"Time   : {now}\n")
        f.write(f"Words  : {len(text.split())}\n\n")
        f.write("Transcription:\n")
        f.write("-" * 55 + "\n")
        f.write(text + "\n")
    pr("saved to transcription_output.txt", "green")


def main():
    pr("\nSpeech Recognition System", "cyan", bold=True)
    pr("Codetech Internship | AI Domain\n", "cyan")

    pr("what do you want to do?", "yellow", bold=True)
    pr("  [1] use microphone", "white")
    pr("  [2] load a wav file", "white")
    pr("  [3] quit\n", "white")

    ch = input("enter 1, 2 or 3: ").strip()

    if ch == "1":
        audio, r = use_mic()
        src = "Microphone"

    elif ch == "2":
        pr("\npaste the full path of your wav file:", "yellow")
        path = input("path: ").strip().strip('"')
        audio, r = use_file(path)
        src = os.path.basename(path) if path else "file"

    elif ch == "3":
        pr("bye!", "cyan")
        return
    else:
        pr("wrong input, try again", "red")
        return

    if audio is None:
        return

    pr("\nconverting speech to text...", "blue")
    text = get_text(audio, r)

    if text:
        show_output(text, src)
        sv = input("save to file? (y/n): ").strip().lower()
        if sv == "y":
            save_it(text, src)
    else:
        pr("\ncouldn't understand the audio", "red")
        pr("try speaking more clearly or reduce background noise", "yellow")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
        sys.exit(0)
