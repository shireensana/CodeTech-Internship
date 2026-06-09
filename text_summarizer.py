import re
import sys
import string
import nltk

from collections import defaultdict

def setup():
    for pkg in ['punkt', 'stopwords', 'punkt_tab']:
        try:
            nltk.download(pkg, quiet=True)
        except:
            pass

setup()

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    has_color = True
except ImportError:
    has_color = False


def cprint(text, color="white", bold=False):
    if not has_color:
        print(text)
        return
    colors = {
        "cyan": Fore.CYAN,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "red": Fore.RED,
        "blue": Fore.LIGHTBLUE_EX,
        "white": Fore.WHITE,
        "magenta": Fore.MAGENTA,
    }
    c = colors.get(color, Fore.WHITE)
    b = Style.BRIGHT if bold else ""
    print(f"{b}{c}{text}{Style.RESET_ALL}")


def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_word_scores(sentences):
    stop_words = set(stopwords.words('english'))
    freq = defaultdict(int)
    for sent in sentences:
        words = word_tokenize(sent.lower())
        for word in words:
            if word not in stop_words and word not in string.punctuation:
                freq[word] += 1
    if freq:
        max_f = max(freq.values())
        for w in freq:
            freq[w] /= max_f
    return freq


def get_sentence_scores(sentences, word_scores):
    scores = {}
    for sent in sentences:
        words = word_tokenize(sent.lower())
        count = len(words)
        if count == 0:
            continue
        total = sum(word_scores.get(w, 0) for w in words)
        scores[sent] = total / count
    return scores


def get_top_sentences(sentences, scores, n):
    ranked = sorted(scores, key=scores.get, reverse=True)
    top = ranked[:n]
    return [s for s in sentences if s in top]


def summarize(text, ratio=0.3):
    text = clean_text(text)
    sentences = sent_tokenize(text)
    total = len(sentences)
    if total == 0:
        return "No sentences found.", 0, 0
    if total <= 2:
        return text, total, total
    n = max(2, min(10, round(total * ratio)))
    word_scores = get_word_scores(sentences)
    sent_scores = get_sentence_scores(sentences, word_scores)
    top = get_top_sentences(sentences, sent_scores, n)
    return ' '.join(top), total, len(top)


def show_results(original, summary, orig_n, summ_n):
    orig_words = len(original.split())
    summ_words = len(summary.split())
    compression = round((1 - summ_n / orig_n) * 100) if orig_n > 0 else 0

    print()
    cprint("=" * 65, "cyan", bold=True)
    cprint("        TEXT SUMMARIZATION TOOL", "cyan", bold=True)
    cprint("        Codetech Internship - Task 1", "cyan")
    cprint("=" * 65, "cyan", bold=True)
    print()
    cprint("ORIGINAL TEXT:", "yellow", bold=True)
    cprint("-" * 65, "yellow")
    print(original)
    print()
    cprint("STATS:", "blue", bold=True)
    cprint(f"   Total Sentences : {orig_n}", "white")
    cprint(f"   Total Words     : {orig_words}", "white")
    print()
    cprint("SUMMARY:", "green", bold=True)
    cprint("-" * 65, "green")
    print(summary)
    print()
    cprint("SUMMARY STATS:", "magenta", bold=True)
    cprint(f"   Summary Sentences : {summ_n}", "white")
    cprint(f"   Summary Words     : {summ_words}", "white")
    cprint(f"   Compression       : {compression}% shorter", "white")
    print()
    cprint("=" * 65, "cyan", bold=True)
    print()


sample_texts = {
    "1": {
        "title": "Artificial Intelligence",
        "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term artificial intelligence had previously been used to describe machines that mimic and display human cognitive skills associated with the human mind, such as learning and problem-solving. This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated. AI applications include advanced web search engines, recommendation systems used by YouTube, Amazon and Netflix, understanding human speech such as Siri and Alexa, self-driving cars such as Waymo, generative or creative tools such as ChatGPT and AI art, automated decision-making, and competing at the highest level in strategic game systems, such as chess and Go. As machines become increasingly capable, tasks considered to require intelligence are often removed from the definition of AI, a phenomenon known as the AI effect. Modern machine learning techniques are the core of many modern AI systems, including the most cutting-edge systems in deep learning and reinforcement learning. The field was founded on the assumption that human intelligence can be so precisely described that a machine can be made to simulate it. This raises philosophical arguments about the mind and the ethics of creating artificial beings endowed with human-like intelligence."
    },
    "2": {
        "title": "Climate Change",
        "text": "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, such as through variations in the solar cycle. But since the 1800s, human activities have been the main driver of climate change, primarily due to burning fossil fuels like coal, oil and gas. Burning fossil fuels generates greenhouse gas emissions that act like a blanket wrapped around the Earth, trapping the sun's heat and raising temperatures. Examples of greenhouse gas emissions that are causing climate change include carbon dioxide and methane. These come from using gasoline for driving a car or coal for heating a building. Clearing land and forests can also release carbon dioxide. Landfills for garbage are a major source of methane emissions. Energy, food, industry, transport, buildings and land use are among the main emitters. Climate change is causing a wide range of negative impacts, including rising sea levels, melting glaciers, more intense storms, droughts, and heat waves. These changes are affecting ecosystems, wildlife, and human communities around the world. Renewable energy sources like solar and wind power are being developed rapidly as alternatives to fossil fuels. International agreements like the Paris Agreement aim to limit global warming to 1.5 degrees Celsius above pre-industrial levels."
    }
}


def run():
    cprint("\nText Summarization Tool", "cyan", bold=True)
    cprint("Codetech Internship | AI Domain\n", "cyan")

    cprint("Choose an option:", "yellow", bold=True)
    cprint("  [1] Artificial Intelligence sample", "white")
    cprint("  [2] Climate Change sample", "white")
    cprint("  [3] Type your own text", "white")
    cprint("  [4] Run both samples\n", "white")

    choice = input("Enter choice (1/2/3/4): ").strip()

    if choice in ["1", "2"]:
        data = sample_texts[choice]
        cprint(f"\nUsing: {data['title']}", "green")
        text = data["text"].strip()

    elif choice == "3":
        cprint("\nPaste your text below. Press Enter twice when done.\n", "green")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        text = " ".join(lines).strip()
        if not text:
            cprint("No text entered. Exiting.", "red")
            return

    elif choice == "4":
        for key in ["1", "2"]:
            data = sample_texts[key]
            cprint(f"\nProcessing: {data['title']}", "magenta", bold=True)
            t = data["text"].strip()
            s, on, sn = summarize(t)
            show_results(t, s, on, sn)
        return

    else:
        cprint("Invalid input, using AI sample.", "red")
        text = sample_texts["1"]["text"].strip()

    print()
    cprint("Summary length:", "yellow", bold=True)
    cprint("  [1] Short  (20%)", "white")
    cprint("  [2] Medium (30%)", "white")
    cprint("  [3] Long   (50%)", "white")

    r = input("Enter choice (1/2/3): ").strip()
    ratio = {"1": 0.2, "2": 0.3, "3": 0.5}.get(r, 0.3)

    cprint("\nSummarizing...\n", "blue")
    summary, orig_n, summ_n = summarize(text, ratio=ratio)
    show_results(text, summary, orig_n, summ_n)

    save = input("Save to file? (y/n): ").strip().lower()
    if save == "y":
        fname = "summary_output.txt"
        with open(fname, "w", encoding="utf-8") as f:
            f.write("TEXT SUMMARIZATION TOOL - CODETECH INTERNSHIP\n")
            f.write("=" * 65 + "\n\n")
            f.write("ORIGINAL TEXT:\n")
            f.write("-" * 65 + "\n")
            f.write(text + "\n\n")
            f.write(f"Sentences: {orig_n}  |  Words: {len(text.split())}\n\n")
            f.write("SUMMARY:\n")
            f.write("-" * 65 + "\n")
            f.write(summary + "\n\n")
            f.write(f"Sentences: {summ_n}  |  Words: {len(summary.split())}\n")
        cprint(f"\nSaved to: {fname}", "green", bold=True)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)
