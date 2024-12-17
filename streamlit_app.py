def main():
    st.title("News Classification App")
    st.write("This is a web app to classify news articles into different categories.")
    
    # Load data
    data = {
        "title": ["RBI revises definition of politically-exposed ...", "NDTV Q2 net profit falls 57.4% to Rs 5.55 cror...", 
                  "Akasa Air ‘well capitalised’, can grow much fa...", "India’s current account deficit declines sharp...", 
                  "States borrowing cost soars to 7.68%, highest ...", "India’s Russian oil imports slip in Oct, Saudi...", 
                  "Neelkanth Mishra appointed part-time chairpers...", "Centre issues advisory to social media platfor...", 
                  "Asian shares rise after eased pressure on bond...", "India’s demand for electricity for ACs to exce..."],
        "category": ["business", "business", "business", "business", "business", "business", "business", "business", 
                     "business", "business"]
    }
    
    df = pd.DataFrame(data)
    
    # Display the data
    st.write("## Data")
    st.write(df)
    
    # EDA
    st.write("## Exploratory Data Analysis")
    
    st.write("### Category Distribution")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='category', data=df)
    st.pyplot(plt)
    
    st.write("### Title Length Distribution")
    df['title_length'] = df['title'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['title_length'], bins=20, kde=True)
    st.pyplot(plt)
    
    # Train the model
    st.write("## Model Training")
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['title'])
    y = df['category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    classifier = GaussianNB()
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)
    grid_search.fit(X_train.toarray(), y_train)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    st.write("### Best Hyperparameters")
    st.write(best_params)
    st.write("### Best Cross-Validation Score")
    st.write(best_score)

if __name__ == "__main__":
    main()

