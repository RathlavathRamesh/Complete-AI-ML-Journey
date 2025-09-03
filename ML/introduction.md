# # 📘 Introduction to ML and AI  

# ##  Artificial Intelligence (AI)  
# - **AI** is the broader concept of machines being able to carry out tasks in a way we would consider *“smart.”*  
# - It is the **simulation of human intelligence processes** by machines, especially computer systems.  
# - **Subfields of AI** include:  
#   - Machine Learning  
#   - Natural Language Processing  
#   - Robotics  
#   - Computer Vision  

# ### Types of AI  
# - **Narrow AI**:  
#   Designed to perform specific tasks.  
#   - *Examples*: Virtual assistants (Siri, Alexa), recommendation systems (Netflix, Amazon), image recognition (Google Photos).  

# - **General AI**:  
#   Designed to perform *any* intellectual task a human can do.  
#   - Still a **theoretical concept** (not yet achieved).  

# ---

# ## Introduction to Machine Learning (ML)  
# - **ML** is a **subset of AI** that focuses on algorithms and statistical models that enable computers to perform tasks **without explicit instructions**, relying on **patterns and inference**.  

# ### Key Points  
# - If AI is the **world**, ML is a **subset** inside it.  
# - ML is a **technique to achieve AI**.  
# - It enables computers to **learn without being explicitly programmed**.  
# - **Applications**: Email filtering, speech recognition, computer vision, recommendation systems, etc.  

# ### Types of ML  
# 1. Supervised Learning  
# 2. Unsupervised Learning  
# 3. Reinforcement Learning  

# ---

# ## 🔹 Types of Machine Learning  

# ### 1️Supervised Learning  
# - **Definition**: Trained on a **labeled dataset** (input + output labels).  
# - **Goal**: Learn mapping from inputs → outputs, then predict outputs for new inputs.  
# - **Algorithms**: Linear Regression, Logistic Regression, Decision Trees, Support Vector Machines (SVM).  

# #### a) Classification  
# - **Output Variable**: **Categorical** (limited values).  
# - **Goal**: Assign inputs to one of the predefined categories.  
# - **Examples**: Logistic Regression, Decision Trees, SVM.  

# ##### Types of Classification  
# - **Binary Classification**  
#   - Output has **two values**: e.g., *Yes/No*, *Spam/Not Spam*.  
#   - Algorithms: Logistic Regression, Decision Trees, SVM.  

# - **Multi-class Classification**  
#   - Output has **more than two values**: e.g., *Cat/Dog/Bird*.  
#   - Algorithms: Decision Trees, Random Forests, SVM.  

# ##### Algorithms Used for Classification  
# - **Logistic Regression**: Models probability (0–1) for binary classification.  
# - **Decision Trees**: Splits data into smaller regions; results in a tree structure.  
# - **Support Vector Machines (SVM)**: Finds the best **hyperplane** to separate classes (linear & non-linear).  

# #### b) Regression  
# - **Output Variable**: **Continuous** (any numeric value).  
# - **Goal**: Predict values based on input variables.  
# - **Examples**: Linear Regression, Polynomial Regression, Ridge Regression.  

# ---

# ### 2️ Unsupervised Learning  
# - **Definition**: Trained on an **unlabeled dataset** (no output labels).  
# - **Goal**: Discover **patterns/structure** in the data.  
# - **Algorithms**: K-Means, Hierarchical Clustering, PCA.  

# #### a) Clustering  
# - **Definition**: Group similar data points based on features.  
# - **Examples**: K-Means, Hierarchical Clustering, DBSCAN.  

# #### b) Dimensionality Reduction  
# - **Definition**: Reduce number of features while keeping important info.  
# - **Goal**: Simplify, visualize, analyze.  
# - **Examples**: PCA, t-SNE.  

# ##### Algorithms Used in Unsupervised Learning  
# - **K-Means Clustering**: Assigns points to nearest cluster center.  
# - **Hierarchical Clustering**: Recursively merges/splits clusters.  
# - **PCA (Principal Component Analysis)**: Finds key components that explain variance in data.  

# ---

# ### 3️⃣ Reinforcement Learning  
# - **Definition**: The model learns by **interacting with an environment** and receiving **rewards/penalties**.  
# - **Goal**: Learn a **policy** that maximizes cumulative rewards over time.  
# - **Examples**: Q-Learning, Deep Reinforcement Learning.  

# ---

# ##  Challenges in ML  
# - **Overfitting**: Model learns training data *too well* → poor performance on new data.  
# - **Need for Labeled Data**: Requires large amounts of quality labeled data.  
