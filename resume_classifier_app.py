import joblib
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from lime.lime_text import LimeTextExplainer
import string
import re
import spacy
import unicodedata
import matplotlib.pyplot as plt
from spacy.matcher import PhraseMatcher
from spacy.cli import download

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")


def load_spacy_model(model_name="en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Downloading {model_name}...")
        download(model_name)
        return spacy.load(model_name)


nlp = load_spacy_model()


tfidf = joblib.load("Resume_Classification_Tokenizer_Tfidf.joblib")
xgb = joblib.load("Resume_Classifier_Model_XGB.joblib")


lemmatization = WordNetLemmatizer()

explainer = LimeTextExplainer(class_names=["Python Developer", "Java Developer", "Front End Developer", "Network Administrator", "Project Manager",
           "Security Analyst", "Software Developer", "Systems Administrator", "Web Developer", "Database Administrator"])

def data_cleaning(text):
    processed_text = []
    text = text.lower()

    text = re.sub(r"\d+", '', text)

    tokens = word_tokenize(text)

    tokens_without_punctuations = [word for word in tokens if word not in string.punctuation]

    tokens_without_stopwords = [word for word in tokens_without_punctuations if word not in stopwords.words("english")]

    lemmatized_tokens = [lemmatization.lemmatize(word, pos = "v") for word in tokens_without_stopwords]

    joined_text = " ".join(lemmatized_tokens)

    joined_text = unicodedata.normalize('NFKD', joined_text)
    joined_text = joined_text.encode('ascii', 'ignore').decode('ascii')


    processed_text.append(joined_text)
    
    return processed_text


st.title("Resume Classification and Job Matching Application")
st.write("This is a Resume Classification and Job Matching Application which can be used to classify Resumes into different categories, understand the rationale behind the classification, get a resume matching score with job description and get a list of matching skills")

user_resume_input = st.text_area("Please enter the Resume Content")

st.subheader("Resume Similarity Score with Job Description")
job_description = st.text_area("Please Enter the Job Description")

prediction_button = st.button("Show Prediction Results")

def predict_proba(text):
    vec = tfidf.transform(text)
    return xgb.predict_proba(vec)


if prediction_button:
   cleaned_data = data_cleaning(user_resume_input)

   vec = tfidf.transform(cleaned_data)

   prediction_prob = xgb.predict_proba(vec)[0]
   prediction = prediction_prob.argmax()

   class_labels = {0: "Python Developer", 1: "Java Developer", 2: "FrontEnd Developer", 3: "Network Administrator", 4: "Project Manager",
           5: "Security Analyst", 6: "Software Developer", 7: "Systems Administrator", 8: "Web Developer", 9: "Database Administrator"}
   
   st.subheader("Prediction Results")
   st.write(f"The Resume belongs to {class_labels[prediction]} Category")
   
   fig, ax = plt.subplots()
   ax.bar(class_labels.values(), prediction_prob)
   plt.xticks(rotation = 90)
   plt.ylabel("Confidence Score")
   plt.title("Prediction Confidence Score")
   st.pyplot(fig)
   
   
   
   
   doc1 = nlp(job_description)
   doc2 = nlp(user_resume_input)
   
   similarity_score = doc1.similarity(doc2)
   st.success(f"Similarity Score of Resume with Job Description is {similarity_score: .2f}")
   fig, ax = plt.subplots()
   ax.pie([similarity_score, 1-similarity_score],
            labels=["Matched", "Unmatched"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#4CAF50", "#FF6F61"])
   ax.axis("equal")
   st.pyplot(fig)
   
   skill_list = [
        # Programming Languages  
        "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "Ruby", "Go", "Rust",  
        # Web Development  
        "HTML5", "CSS3", "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring Boot",  
        # Databases  
        "SQL", "PostgreSQL", "MySQL", "MongoDB", "Oracle", "Redis", "Firebase",  
        # DevOps & Cloud  
        "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Terraform", "Jenkins", "Git",  
        # Data Science/AI  
        "Machine Learning", "Deep Learning", "NLP", "TensorFlow", "PyTorch", "Scikit-learn", "Pandas",  
        # Networking & Security  
        "TCP/IP", "DNS", "Firewalls", "VPN", "Kali Linux", "Metasploit", "SIEM", "CISSP",  
        # Systems & Admin  
        "Linux", "Bash", "PowerShell", "Windows Server", "VMware", "Ansible",  
        # Project Management  
        "Agile", "Scrum", "Jira", "Confluence", "Risk Management",  
        # Other Tools  
        "Power BI", "Tableau", "Figma", "Wireshark", "Splunk"  
    ]
   
   st.subheader("Matched Skills Analysis")
   matcher = PhraseMatcher(nlp.vocab)
   patterns = [nlp.make_doc(skill) for skill in skill_list]
   matcher.add("Skills", patterns)
   
   matches = matcher(doc2)
   matched_skills = set()
   
   for match_id, start, end in matches:
        span = doc2[start:end]
        matched_skills.add(span.text)
        
   for i in sorted(matched_skills):
      st.write(f"{i}")


   exp = explainer.explain_instance(user_resume_input, predict_proba, num_features=10, top_labels=3)
   st.subheader("Why this prediction?")
   
   tab1, tab2 = st.tabs(["Main Factors", "Detailed View"])
   
   with tab1:
    st.write("Top positive contributors:")
    for feature, weight in [x for x in exp.as_list(0) if x[1] > 0][:3]:
        st.write(f"✓ {feature}")
        
    st.write("\nTop negative contributors:")
    for feature, weight in [x for x in exp.as_list(0) if x[1] < 0][:3]:
        st.write(f"✗ {feature}")
    
   with tab2:
    for label in exp.top_labels:
        st.write(f"### {class_labels[label]}")
        for feature, weight in exp.as_list(label):
           st.progress(abs(weight), text=f"{feature} ({'+' if weight > 0 else '-'}{abs(weight):.2f})")
    





   
       






