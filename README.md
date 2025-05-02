Course Recommendation System using LLM
This project is a course recommendation system that suggests online courses to users based on their interests.
It leverages OpenAI's text-embedding-ada-002 model to generate text embeddings of course titles and compute similarity with user input.

📌 Features
Generates course recommendations from Udemy course dataset.

Uses LLM embeddings for semantic similarity.

Provides top 10 most relevant courses based on user interests.

🛠️ Tech Stack
Python (pandas, numpy, scikit-learn)

OpenAI API (text-embedding-ada-002)

tiktoken (for token estimation)

📂 Dataset
Dataset: udemy_courses2.csv

Columns used:

title: Course title

headline: Course subject/description

url: Course link

⚙️ How it works
Load Dataset: Reads course data from CSV file.

Generate Embeddings: Creates embeddings for course titles using OpenAI embedding model.

Compute Similarity: Calculates cosine similarity between user query embedding and course embeddings.

Return Recommendations: Displays top 10 recommended courses.

💡 Future Improvements
Include course descriptions in embedding for better recommendations.

Deploy as web app using Streamlit 

🏆 Author
Mohammad Shaheen
AI Engineer | Machine Learning Enthusiast
LinkedIn:
www.linkedin.com/in/mohammad-shaheen-276ba5231
