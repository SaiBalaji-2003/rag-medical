# RAG Medical Q&A System: A Smart Assistant for Understanding Medical Jargon
I created this project to build a helpful assistant that can answer plain-English questions about these complex medical codes. Think of it as a friendly translator for medical jargon, designed to make this information more accessible to everyone.

## How It Works: The Tech Behind the Scenes

I built this system using a combination of powerful, open-source tools that can run on a regular computer. Here’s a peek at the key players and what they do:

- **The Brains of the Operation (AI Models):**
    - To understand the *meaning* of your questions and the medical documents, I used a model called **`all-MiniLM-L6-v2`**. It's great at figuring out which parts of a document are most relevant to what you're asking.
    - To generate a clear, human-like answer, I chose **`DistilGPT-2`**, a smaller and faster cousin of the famous GPT models. It's perfect for running locally without needing a supercomputer.

- **The Digital Filing Cabinet (ChromaDB):**
    - This is where the system stores all the medical information after it's been broken down into understandable chunks. It's a special kind of database that's designed for finding information based on meaning, not just keywords.

- **The User-Friendly Interface (Streamlit):**
    - I used Streamlit to create a simple and clean web interface. This is what lets you upload documents and ask questions without needing to be a tech expert.

- **The Supporting Crew (Other Libraries):**
    - I also used a handful of other essential tools, like **PyTorch** to run the AI models, **pandas** for organizing data, and **Plotly** to create some cool charts and visualizations.

## My Partnership with AI

Building a system like this from scratch is a big undertaking. To help me along the way, I collaborated with AI tools in a few key areas:

1.  **Figuring Out How to "Read" the Documents:** Medical documents are complex. I worked with AI to find the best way to break them down into smaller pieces without losing important context. We settled on a "sentence-aware" approach, which means the system tries to keep related sentences together, making the information more coherent.

2.  **Choosing the Right Tools for the Job:** There are tons of AI models out there, and it can be hard to know which ones to pick. I used AI to research and compare different options, which helped me find the perfect balance between accuracy, speed, and the ability to run on a standard computer.

3.  **Designing a Clean and Simple Interface:** I'm not a professional web designer, so I got some help from AI to lay out the user interface. It gave me great ideas for how to organize the tabs, buttons, and display the results in a way that’s intuitive and easy to use.

In every case, the AI provided suggestions and ideas, but I was the one who made the final decisions, testing and tweaking everything to fit the unique needs of this project.

## The Blueprint: My Design Philosophy

When I was building this system, I had a few key principles in mind:

- **Keep it Simple and Modular:** I designed each part of the system (document processing, AI models, etc.) to be independent. This makes it easier to test, maintain, and upgrade in the future.
- **Local and Private:** I wanted this system to run entirely on your own computer, without needing to send your data to an external server. This ensures privacy and means you don't need an internet connection to use it.
- **Built for a Specific Need:** Rather than trying to build a system that knows everything, I focused on one specific area: ICD-10 mental health classifications. This allowed me to create a more accurate and reliable tool for that specific task.

## Real-World Constraints: What I Couldn't Do (Yet!)

I built this project with a 4-hour time limit and the resources of a standard laptop. That meant I had to make some tough choices and accept a few limitations:

- **Time Was Tight:** With more time, I would have loved to experiment with larger, more powerful AI models, fine-tune them on medical data to improve their accuracy, and build out more advanced features.
- **Limited Resources:** Since I was building this to run on a regular computer, I had to choose smaller, CPU-friendly models. This means it might not be as fast or as accurate as a system running on a high-end server with a powerful GPU.
- **Keeping it Focused:** To get a working prototype done quickly, I had to keep the scope narrow. The system only works with plain text documents in English, and it doesn't have features like user accounts or advanced analytics.

## What I Learned

Despite the limitations, this project was a huge success. It's a fully functional proof-of-concept that shows how powerful and accessible AI can be. It's a great starting point for a tool that could one day help students, researchers, and even patients make sense of the complex world of medical information.

Thanks for checking it out!