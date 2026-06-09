#INTERNSHIP TASKS CODTECH ARTIFICIAL INTELLIGENCE

COMPANY : CODTECH IT SOLUTIONS

*NAME* : SHIREEN SANA

*INTERN ID : CTIS05S62

DOMAIN : ARTIFICIAL INTELLIGENCE

DURATION : 4 WEEKS

*MENTOR* : NEELA SANTHOSH

DESCRIPTION :

            CODETECH INTERNSHIP - AI DOMAIN
PROJECT DESCRIPTION
Name: Shireen Sana
Domain: Artificial Intelligence
Internship Provider: Codetech IT Solutions
=============================================================


TASK 1 - TEXT SUMMARIZATION TOOL
=============================================================

For my first task I had to build a text summarization tool that could take a long article
or paragraph and give back a short summary of it. Honestly when I first read the task I
was a little confused about where to even start because I hadn't really worked with NLP
before. After doing some research I found that there are mainly two types of summarization
- extractive and abstractive. Extractive basically picks the most important sentences from
the original text while abstractive generates completely new sentences. I decided to go
with extractive summarization because it was more beginner friendly and didn't require
a GPU or huge model downloads.

For this task I used Python as my programming language and the main library I used was
NLTK which stands for Natural Language Toolkit. NLTK is a really popular library for
working with text data in Python. I also used the colorama library just to make the
terminal output look nicer with colors.

The way my tool works is pretty straightforward. First it takes the input text and cleans
it by removing extra spaces and special characters. Then it uses NLTK's sent_tokenize
function to split the text into individual sentences. After that it calculates a score
for each word based on how frequently it appears in the text while ignoring common words
like "the", "is", "and" which are called stop words. Once every word has a score the
tool scores each sentence by adding up the scores of all the words in it. Finally it
picks the top scoring sentences and puts them together to form the summary.

I also added a feature where the user can choose how detailed they want the summary to be
- short, medium or detailed. The tool shows the original text, the summary, total word
count and how much the text was compressed. I also added an option to save the results
to a text file which I thought was a nice touch.

Running this for the first time gave me a ModuleNotFoundError for NLTK which I fixed by
running pip install nltk in the terminal. After that it worked perfectly and I tested it
on topics like artificial intelligence and climate change.

Tools and Libraries used in Task 1:
- Python 3
- NLTK (Natural Language Toolkit) for tokenization and stop words
- Colorama for colored terminal output
- Regular expressions (re module) for text cleaning


TASK 2 - SPEECH RECOGNITION SYSTEM
=============================================================

The second task was to build a speech recognition system that could transcribe short audio
clips. This one was really exciting for me because I had never done anything with audio
or speech before. The idea of making a computer understand what I'm saying felt really
cool.

For this task I used the SpeechRecognition library in Python which is honestly one of the
easiest ways to get started with speech recognition. Under the hood it uses Google's
Speech Recognition API to convert speech to text which is the same technology used in
apps like Google Assistant. I also used the PyAudio library to access the microphone
and capture audio in real time.

The system I built has two modes. The first mode lets you record directly from the
microphone. When you choose this option the program first adjusts for background noise
for about two seconds which really helps with accuracy especially if you're in a noisy
environment. Then it listens for your speech and sends the audio to Google's API to
get back the transcribed text. The second mode lets you load a wav audio file from
your computer and transcribe it which is useful if you already have a recorded clip.

I ran into a few issues while building this. PyAudio was a bit tricky to install on
Windows and I had to look up how to install it properly. Also the first time I ran the
script it was in the OUTPUT tab of VS Code which is read only so I couldn't type my
input. I figured out that I needed to switch to the TERMINAL tab to actually interact
with the program.

The output shows the transcribed text along with the source (microphone or file), the
timestamp and the word count. There's also an option to save the transcription to a
text file called transcription_output.txt.

Tools and Libraries used in Task 2:
- Python 3
- SpeechRecognition library for converting speech to text
- PyAudio for microphone input and audio capture
- Google Speech Recognition API (free, used via SpeechRecognition)
- Colorama for colored terminal output
- datetime module for timestamps


TASK 3 - NEURAL STYLE TRANSFER
=============================================================

Task 3 was by far the most advanced one and it took me the most time to understand and
implement. The goal was to apply the artistic style of one image onto another image using
deep learning. For example taking a photo and making it look like it was painted in the
style of Van Gogh's Starry Night.

The technique used here is called Neural Style Transfer and it was first introduced in a
research paper by Gatys et al. in 2015. The basic idea is that a deep neural network can
separate the content of an image from its style. Content refers to the objects and
structures in the image while style refers to the textures, colors and brushstroke patterns.

For this task I used PyTorch which is one of the most popular deep learning frameworks.
I also used a pretrained model called VGG19 which is a convolutional neural network that
was originally trained on millions of images for image classification. The reason we use
VGG19 here is because its intermediate layers have learned to detect features like edges,
textures and patterns which is exactly what we need for style transfer.

The way the algorithm works is it starts with the content image and then gradually modifies
it step by step using an optimizer called LBFGS to make it match the style of the style
image. It measures content loss by comparing the content features of the output image with
the content image, and style loss by comparing the Gram matrices of the style features.
The Gram matrix captures the correlation between different feature maps which represents
the texture and style information.

I used torchvision to load and preprocess the images and PIL (Pillow) for saving the
output image. The training ran for 300 steps and took about 64 seconds on my CPU which
I was honestly surprised by because I expected it to take much longer. The output image
was saved as styled_output.png and when I opened it I could clearly see the artistic
style applied to the content image.

Tools and Libraries used in Task 3:
- Python 3
- PyTorch for building and running the neural network
- Torchvision for pretrained VGG19 model and image transforms
- PIL / Pillow for image loading and saving
- Requests for downloading sample images
- Colorama for colored terminal output
- VGG19 pretrained model (loaded from PyTorch model hub)


TASK 4 - TEXT GENERATION MODEL
=============================================================

The fourth and final task was to build a text generation model using either GPT or LSTM
to generate coherent paragraphs on specific topics. Text generation is one of the most
interesting applications of deep learning because it involves teaching a machine to write
like a human.

For this task I implemented two approaches. The first approach uses GPT-2 which is a
powerful language model developed by OpenAI and available for free through the
HuggingFace Transformers library. GPT-2 was trained on a massive amount of internet
text and can generate very coherent and natural sounding paragraphs given a starting
prompt.

The second approach I implemented is a Markov Chain model which works similarly to how
LSTM models work in terms of predicting the next word based on previous words. I trained
this model on my own text data about different topics. The model learns which words tend
to follow other words and uses that knowledge to generate new sentences. While it's not
as sophisticated as GPT-2 it works completely offline without needing any internet
connection or heavy computation.

My tool lets the user choose from five different topics including Artificial Intelligence,
Climate Change, Technology, Space Exploration and Health. The user can also enter their
own custom topic. After choosing a topic the user enters a starting prompt like "Artificial
intelligence is changing" and the model continues writing from there. The user can also
choose the length of the generated text - short, medium or long.

I added a typewriter effect to the output which prints the text character by character
which I thought made it look really cool. The output also shows which model was used,
the topic, the prompt and the word count. Results can be saved to a text file called
generated_text_output.txt.

One challenge I faced was installing the Transformers library because it has a lot of
dependencies and took some time to download. But once it was installed GPT-2 worked
really well and generated impressive text.

Tools and Libraries used in Task 4:
- Python 3
- HuggingFace Transformers library for GPT-2 model
- GPT-2 pretrained language model by OpenAI
- Custom Markov Chain implementation for offline generation
- Colorama for colored terminal output
- Random and time modules for generation logic


OVERALL SUMMARY
=============================================================

Doing these four tasks taught me a lot about the field of Artificial Intelligence and
how different techniques can be applied to solve real world problems. I got hands on
experience with Natural Language Processing, Speech Recognition, Computer Vision and
Text Generation which are four of the most important areas in modern AI.

I also got much more comfortable with Python and learned how to install and use libraries
like NLTK, SpeechRecognition, PyTorch and HuggingFace Transformers. Debugging errors
like ModuleNotFoundError and UnicodeEncodeError taught me how to read error messages
and find solutions which is a really important skill for any developer.

Overall this internship was a great learning experience and I feel much more confident
working with AI and Python projects now than I did when I started.    
