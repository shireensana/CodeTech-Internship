import sys
import re
import random
import time

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    col = True
except ImportError:
    col = False


def pr(text, color="white", bold=False):
    if not col:
        print(text)
        return
    c = {
        "cyan": Fore.CYAN, "green": Fore.GREEN,
        "yellow": Fore.YELLOW, "red": Fore.RED,
        "blue": Fore.LIGHTBLUE_EX, "white": Fore.WHITE,
        "magenta": Fore.MAGENTA
    }.get(color, Fore.WHITE)
    b = Style.BRIGHT if bold else ""
    print(f"{b}{c}{text}{Style.RESET_ALL}")


# try to use GPT-2 if transformers is installed
# falls back to LSTM-style markov chain if not
def check_transformers():
    try:
        from transformers import pipeline
        return True
    except ImportError:
        return False


# GPT-2 based generation
def generate_with_gpt2(prompt, max_length=200, temperature=0.9):
    pr("Loading GPT-2 model... (first time takes ~1 min to download)", "blue")
    from transformers import pipeline
    generator = pipeline("text-generation", model="gpt2")
    pr("Model loaded! Generating text...\n", "green")

    result = generator(
        prompt,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=50256
    )
    return result[0]["generated_text"]


# LSTM-style Markov Chain text generation (no internet / no GPU needed)
class MarkovChain:
    def __init__(self, order=2):
        self.order = order
        self.chain = {}

    def train(self, text):
        words = text.split()
        for i in range(len(words) - self.order):
            key = tuple(words[i:i + self.order])
            next_word = words[i + self.order]
            if key not in self.chain:
                self.chain[key] = []
            self.chain[key].append(next_word)

    def generate(self, seed_words, length=100):
        words = seed_words.split()
        if len(words) < self.order:
            words = words + ["the"] * (self.order - len(words))

        result = list(words)
        current = tuple(words[-self.order:])

        for _ in range(length):
            if current in self.chain:
                next_word = random.choice(self.chain[current])
                result.append(next_word)
                current = tuple(result[-self.order:])
            else:
                # pick a random key that starts similarly
                keys = list(self.chain.keys())
                if keys:
                    current = random.choice(keys)
                else:
                    break

        return " ".join(result)


# training data for different topics
TOPIC_DATA = {
    "artificial intelligence": """
    Artificial intelligence is transforming the world in remarkable ways. Machine learning algorithms
    can now process vast amounts of data to identify patterns and make predictions. Deep learning
    neural networks have achieved superhuman performance in image recognition and natural language
    processing tasks. AI systems are being deployed in healthcare to diagnose diseases, in finance
    to detect fraud, and in transportation to power self-driving vehicles. The development of large
    language models has enabled machines to generate coherent text, answer questions, and engage
    in complex reasoning. Researchers are working to make AI systems more transparent, fair, and
    aligned with human values. The future of artificial intelligence holds enormous promise for
    solving some of humanity's greatest challenges including climate change, drug discovery, and
    education. However it also raises important ethical questions about privacy, job displacement
    and the concentration of power. Building AI that benefits all of humanity requires collaboration
    between technologists, policymakers, and society at large. Neural networks learn by adjusting
    millions of parameters through a process called backpropagation. Reinforcement learning allows
    agents to learn optimal behavior through trial and error in simulated environments. Computer
    vision systems can now detect objects in images with greater accuracy than human experts.
    Natural language processing has enabled machines to translate between hundreds of languages
    in real time. AI assistants are becoming increasingly capable of understanding context and
    carrying on meaningful conversations with humans.
    """,

    "climate change": """
    Climate change is one of the most pressing challenges facing humanity today. Rising global
    temperatures are causing ice caps to melt and sea levels to rise at an alarming rate. Extreme
    weather events such as hurricanes floods and droughts are becoming more frequent and intense.
    Scientists have concluded that human activities particularly the burning of fossil fuels are
    the primary driver of climate change. Carbon dioxide and other greenhouse gases trap heat in
    the atmosphere creating a warming effect that disrupts natural climate patterns. Renewable
    energy sources such as solar and wind power offer promising alternatives to fossil fuels.
    Transitioning to clean energy could significantly reduce greenhouse gas emissions and slow
    the pace of global warming. Governments around the world are implementing policies to reduce
    carbon emissions and promote sustainable development. Individual actions such as reducing
    energy consumption and choosing plant-based diets can also contribute to climate solutions.
    Technology innovation in areas like energy storage and carbon capture may play a crucial role
    in addressing the climate crisis. International cooperation is essential for tackling a global
    problem that knows no borders. The impacts of climate change are felt most severely by
    vulnerable communities in developing nations who have contributed least to the problem.
    Protecting forests and restoring ecosystems can help absorb carbon dioxide from the atmosphere.
    The transition to a sustainable economy presents opportunities for green jobs and economic growth.
    """,

    "technology": """
    Technology is advancing at an unprecedented pace transforming every aspect of human life and
    society. Smartphones have put powerful computers in the pockets of billions of people around
    the world. The internet has connected humanity and enabled the free flow of information and
    ideas across borders. Cloud computing allows individuals and organizations to access powerful
    computing resources on demand. Artificial intelligence and machine learning are automating
    repetitive tasks and augmenting human capabilities in profound ways. The Internet of Things
    is connecting everyday objects to the internet creating smart homes and cities. Blockchain
    technology enables secure decentralized record keeping without the need for trusted intermediaries.
    Virtual reality and augmented reality are creating immersive digital experiences that blur the
    line between physical and digital worlds. Advances in biotechnology are revolutionizing medicine
    enabling personalized treatments and potentially curing genetic diseases. Quantum computing
    promises to solve problems that are impossible for classical computers to tackle. The rapid
    pace of technological change creates both enormous opportunities and significant challenges
    for society. Cybersecurity has become increasingly important as more of our lives move online.
    Digital literacy is becoming an essential skill for participation in the modern economy.
    Technology companies have accumulated enormous power and influence raising concerns about
    monopoly and the need for stronger regulation.
    """,

    "space exploration": """
    Space exploration has captured the human imagination since the dawn of the space age. The
    Apollo missions demonstrated that humans could travel to the moon and return safely to Earth.
    Robotic spacecraft have explored every planet in our solar system sending back stunning images
    and scientific data. The Hubble Space Telescope has revealed the beauty and vastness of the
    universe capturing images of galaxies billions of light years away. Mars rovers have discovered
    evidence that the red planet once had liquid water on its surface raising hopes for finding
    ancient microbial life. Private companies like SpaceX and Blue Origin are developing reusable
    rockets that could dramatically reduce the cost of reaching orbit. NASA plans to return humans
    to the moon and eventually send astronauts to Mars in the coming decades. The search for
    exoplanets has revealed thousands of worlds orbiting distant stars some of which may be
    habitable. Space telescopes like the James Webb Space Telescope are peering back to the dawn
    of the universe to understand how galaxies and stars formed. International cooperation on
    projects like the International Space Station has shown that nations can work together in space
    despite political differences on Earth. The colonization of other planets could eventually
    make humanity a multi-planetary species ensuring the long-term survival of civilization.
    """,

    "health": """
    Good health is the foundation of a happy and productive life. Regular physical exercise
    strengthens the heart muscles and bones while improving mental health and cognitive function.
    A balanced diet rich in fruits vegetables whole grains and lean proteins provides the nutrients
    the body needs to function optimally. Adequate sleep is essential for physical and mental
    recovery allowing the body to repair itself and the mind to consolidate memories. Stress
    management techniques such as meditation yoga and deep breathing can reduce the harmful
    effects of chronic stress on the body. Regular medical checkups allow doctors to detect and
    treat health problems before they become serious. Vaccines have been one of the greatest
    public health achievements preventing millions of deaths from infectious diseases. Mental
    health is just as important as physical health and seeking help for depression and anxiety
    is a sign of strength not weakness. Social connections and strong relationships contribute
    significantly to both mental and physical wellbeing. Avoiding smoking excessive alcohol and
    recreational drugs can prevent numerous health problems and extend lifespan. Advances in
    medical technology are enabling earlier diagnosis and more effective treatment of diseases
    like cancer and heart disease. Personalized medicine tailored to an individuals genetic
    makeup promises to revolutionize healthcare in the coming decades.
    """
}


def typewriter_effect(text, delay=0.01):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()


def show_header():
    print()
    pr("=" * 60, "cyan", bold=True)
    pr("        GENERATIVE TEXT MODEL", "cyan", bold=True)
    pr("        Codetech Internship - Task 4", "cyan")
    pr("=" * 60, "cyan", bold=True)
    print()


def show_result(prompt, generated_text, method, topic):
    print()
    pr("=" * 60, "cyan", bold=True)
    pr("  GENERATION COMPLETE", "green", bold=True)
    pr("=" * 60, "cyan", bold=True)
    print()
    pr(f"  Topic    : {topic.title()}", "blue")
    pr(f"  Method   : {method}", "blue")
    pr(f"  Prompt   : {prompt}", "blue")
    pr(f"  Words    : {len(generated_text.split())}", "blue")
    print()
    pr("GENERATED TEXT:", "yellow", bold=True)
    pr("-" * 60, "yellow")
    print()
    typewriter_effect(generated_text, delay=0.01)
    print()
    pr("=" * 60, "cyan", bold=True)
    print()


def save_output(prompt, text, topic, method):
    fname = "generated_text_output.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("GENERATIVE TEXT MODEL - CODETECH INTERNSHIP\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Topic  : {topic.title()}\n")
        f.write(f"Method : {method}\n")
        f.write(f"Prompt : {prompt}\n")
        f.write(f"Words  : {len(text.split())}\n\n")
        f.write("GENERATED TEXT:\n")
        f.write("-" * 60 + "\n")
        f.write(text + "\n")
    pr(f"\nSaved to: {fname}", "green", bold=True)


def run():
    show_header()

    has_transformers = check_transformers()

    if has_transformers:
        pr("GPT-2 (transformers) is available!", "green")
        pr("Choose generation method:", "yellow", bold=True)
        pr("  [1] GPT-2  - AI powered, very coherent text", "white")
        pr("  [2] Markov - Fast, no internet needed\n", "white")
        method_choice = input("Enter choice (1/2): ").strip()
        use_gpt2 = method_choice == "1"
    else:
        pr("Transformers not installed. Using Markov Chain model.", "yellow")
        pr("To use GPT-2 run: pip install transformers\n", "blue")
        use_gpt2 = False

    print()
    pr("Choose a topic:", "yellow", bold=True)
    pr("  [1] Artificial Intelligence", "white")
    pr("  [2] Climate Change", "white")
    pr("  [3] Technology", "white")
    pr("  [4] Space Exploration", "white")
    pr("  [5] Health", "white")
    pr("  [6] Enter custom topic\n", "white")

    topic_map = {
        "1": "artificial intelligence",
        "2": "climate change",
        "3": "technology",
        "4": "space exploration",
        "5": "health"
    }

    tc = input("Enter choice (1-6): ").strip()

    if tc in topic_map:
        topic = topic_map[tc]
    elif tc == "6":
        topic = input("Enter your topic: ").strip().lower()
        if topic not in TOPIC_DATA:
            pr(f"No training data for '{topic}'. Using 'technology' instead.", "yellow")
            topic = "technology"
    else:
        pr("Invalid choice, using Artificial Intelligence.", "red")
        topic = "artificial intelligence"

    print()
    pr(f"Topic selected: {topic.title()}", "green")
    print()
    pr("Enter a prompt to start the text generation:", "yellow", bold=True)
    pr("Example: 'Artificial intelligence is'", "white")
    pr("Example: 'The future of technology'", "white")
    pr("Example: 'Climate change affects'\n", "white")

    prompt = input("Your prompt: ").strip()
    if not prompt:
        prompt = topic.title() + " is"
        pr(f"No prompt entered. Using: '{prompt}'", "yellow")

    print()
    pr("How long should the generated text be?", "yellow", bold=True)
    pr("  [1] Short  - ~50 words", "white")
    pr("  [2] Medium - ~100 words", "white")
    pr("  [3] Long   - ~200 words\n", "white")

    lc = input("Enter choice (1/2/3): ").strip()
    length = {"1": 50, "2": 100, "3": 200}.get(lc, 100)

    print()
    pr("Generating text...\n", "blue")

    if use_gpt2:
        try:
            generated = generate_with_gpt2(prompt, max_length=length * 5)
            method = "GPT-2 (Transformers)"
        except Exception as e:
            pr(f"GPT-2 failed: {e}", "red")
            pr("Falling back to Markov Chain...", "yellow")
            use_gpt2 = False

    if not use_gpt2:
        training_text = TOPIC_DATA.get(topic, TOPIC_DATA["technology"])
        model = MarkovChain(order=2)
        model.train(training_text)
        generated = model.generate(prompt, length=length)
        method = "Markov Chain (LSTM-style)"

    show_result(prompt, generated, method, topic)

    save = input("Save output to file? (y/n): ").strip().lower()
    if save == "y":
        save_output(prompt, generated, topic, method)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)
