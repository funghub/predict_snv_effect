import pandas as pd
import re
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, auc
from sklearn import metrics

def initial_training():

    print('\n---------- Initial Model Training and Evaluation ----------')

    '''
    1. Cleaning the dataset and preparing it for the model training
    '''
    # import the csv file into a pandas dataframe
    df_data = pd.read_csv("data.csv") # make sure your cd is the folder for this project

    # clean the dataset to remove chr# and empty rows
    df_data = df_data.dropna()

    # insert a column to state which chromosome for each gene
    df_data.loc[df_data["Gene"]=="TYR", "chr"] = str(11)
    df_data.loc[df_data["Gene"]=="CDKN2A", "chr"] = str(9)
    df_data.loc[df_data["Gene"]=="SLC45A2", "chr"] = str(5)
    df_data.loc[df_data["Gene"]=="MC1R", "chr"] = str(16) # use this for test data set

    # change column names and remove 'consequence'
    df_data.rename(columns={'Gene':'gene', 'Intron SNV':'rs', 'Position':'pos',
        'Ref_allele':'ref', 'Alt_allele':'alt',
        'Upstream(25bp0':'upstream_seq',
        'downstream(25bp)':'downstream_seq', 'Positive(1) or negative(0)':'label_0-1'}, inplace=True)
    df_data = df_data[['gene', 'chr','pos','rs', 'distance_to_tss', 'is_promoter', 'is_enhancer', 'ref', 'alt', 'Histone data', 'TF data', 'upstream_seq','downstream_seq','label_0-1']]


    def extract_score(score_name, data_type):
        '''
        Extract the histone modification/TF data and create new columns for each term
        '''
        pattern = rf'{score_name}\s*=\s*(\d+\.\d+)'
        for x, scores in df_data[data_type].items():
            match_score = re.findall(pattern, scores)
            if match_score:
                df_data.loc[x, score_name] = match_score[0]
            else:
                df_data.loc[x, score_name] = str(0)
            # else must be 0 because logistic regression can only take numeric features
            
    # extract the histone modification and create new columns
    extract_score("H3K27me3", "Histone data")
    extract_score("H3K79me2", "Histone data")
    extract_score("H3K36me3", "Histone data")
    extract_score("H3K9me3", "Histone data")

    # extract the TF and create new columns
    extract_score("E2F4", "TF data")
    extract_score("MYC", "TF data")
    extract_score("EZH2", "TF data")
    extract_score("CBX8", "TF data")
    extract_score("SUZ12", "TF data")
    extract_score("ESR1", "TF data")
    extract_score("TCF12", "TF data")
    extract_score("FOSL1", "TF data")
    extract_score("GABPA", "TF data")

    # remove Histone data and TF data, and replace with all the data titles we have now
    df_data = df_data[['gene', 'chr','pos','rs', 
                    'distance_to_tss', 'is_promoter', 
                    'is_enhancer', 'ref', 'alt', 
                    'H3K36me3', 'H3K79me2', 'H3K27me3', 'H3K9me3', # Histone Modification
                    'E2F4','MYC','EZH2','CBX8','SUZ12','ESR1', # TF
                    'TCF12','FOSL1','GABPA', # TF
                    'upstream_seq','downstream_seq',
                    'label_0-1']]


    def calculate_GC(sequence: str) -> float:
        '''
        Calculate the GC content
        '''
        sequence = sequence.replace(" ","")
        # print(sequence) # test to see if remove spaces
        num_G = sequence.count("G")
        num_C = sequence.count("C")
        sum_GC = num_G + num_C
        return sum_GC / len(sequence)

    # Enter the GC content fraction out of total bp for upstream and downstream
    df_data["GC_content_up"] = df_data["downstream_seq"].apply(calculate_GC)
    df_data["GC_content_down"] = df_data["upstream_seq"].apply(calculate_GC)

    # Extract training dataset
    df_training = df_data[df_data["gene"].isin(["TYR", "SLC45A2", "CDKN2A"])]
    # print(df_training) # validate if training data set is made

    # Extract MC1R dataset
    df_MC1R = df_data[df_data["gene"]=="MC1R"]
    # print(df_MC1R) # validate if data set is made

    # print(df_training.head(10)) # see the first 10 entries into data
    # print(df_training.columns) # check to see what columns present


    '''
    2. Building the Logistic Regression Model
    modified from source: https://www.datacamp.com/tutorial/understanding-logistic-regression-python
    '''
    # Selecting the features (split dataset features & target variables) for training set
    feature_cols = ['distance_to_tss', 'is_promoter',
        'is_enhancer', 'ref', 'alt', 'H3K36me3', 'H3K79me2', 'H3K27me3',
        'H3K9me3', 'E2F4', 'MYC', 'EZH2', 'CBX8', 'SUZ12', 'ESR1', 'TCF12',
        'FOSL1', 'GABPA', 'GC_content_up', 'GC_content_down']
    # drop(columns=['gene', 'chr', 'pos', 'rs', 'upstream_seq', 'downstream_seq','label'])
    X = df_training[feature_cols]
    Y = df_training["label_0-1"]

    # ref and alt to one-hot encoding representing nucleotide A,C,G,T
    X,Y = cleaning_XY(X,Y)

    # split the data into training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=21)

    # create a model pipeline to run to standardize dataset 
    # https://www.geeksforgeeks.org/machine-learning/what-is-exactly-sklearnpipelinepipeline/
    # Developing the model by instantiation
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=21))# random state for reproducibility seeding
    ])

    # Fit the model with the training data
    model = model.fit(X_train, y_train)

    # before merging, make sure epigenetic features are float
    df_training = df_training.astype({'H3K36me3':float, 'H3K79me2':float, 
                                    'H3K27me3':float, 'H3K9me3':float, 
                                    'E2F4':float, 'MYC':float, 'EZH2':float, 
                                    'CBX8':float, 'SUZ12':float, 'ESR1':float, 
                                    'TCF12':float,'FOSL1':float, 'GABPA':float})

    # merge test dataset with original training dataset to see which intron snv
    X_test_merge = pd.merge(X_test, df_training, 
            on = None, #['distance_to_tss', 'is_promoter','is_enhancer'],
            how = "left")
    X_test_merge = X_test_merge[['gene', 'rs', 'chr', 'pos', 'ref', 
                                'alt', 'label_0-1']]

    # print("Given test dataset to model merged with true labels for reference:\n", X_test_merge)
    # print("\nTrue labels for test dataset:\n", y_test.to_numpy()) # true labels or use .values

    # predicted labels from X_test inputted into model: pathogenic=1 or benign=0
    y_pred = model.predict(X_test)
    # print("Predicted labels for test dataset:\n", y_pred) # predicted labels


    # Present Finalized df comparing predictions and true labels
    X_test_merge["pred_label"] = y_pred # add predicted labels to table

    # predict the probabilities/likelihood of these predictions
    y_score = model.predict_proba(X_test)
    # print(f"""\nScores for each predicted label ({model.classes_}) 
    # in each row of dataset:\n""", y_score) # likelihood of labels
    # add these scores to X_test_merge df
    X_test_merge["Likelihood_Pathogenic/Uncertain Sig"] = y_score[:,1]

    X_test_merge.rename(columns={"label_0-1":"true_label", "rs": "Intron_SNV"}, inplace=True) # change colname for true labels
    print("""\nComparing true labels with predicted labels, 
    including the predicted likelihood of pathogenicity/uncertain significance (Initial Training):\n""", X_test_merge)


    '''
    3. Evaluating the Model - Performance Analysis
    https://www.geeksforgeeks.org/machine-learning/ml-logistic-regression-using-python/
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    '''
    # find the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred) # (y_true, y_pred)
    # print("Accuracy: {:.2f}%".format(accuracy*100))
    print(f"\nAccuracy (Initial Training): {accuracy:.2%}")

    # confusion matrix
    # ith row and jth column (TN 0,0; FN 1,0; TP 1,1; FP 0,1)
    cf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (Initial Training):\n", cf_matrix)

    # plot confusion matrix: "Visualizing confusion matrix using a heatmap"
    # https://www.datacamp.com/tutorial/understanding-logistic-regression-python
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = ["Benign", "Pathogenic/Uncertain"]
    fig, ax = plt.subplots()
    num_ticks = np.arange(len(labels))
    plt.xticks(num_ticks, labels)
    plt.yticks(num_ticks, labels)

    # generate the heatmap
    sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap="coolwarm", fmt="g")
    ax.xaxis.set_label_position("bottom")
    plt.tight_layout()
    plt.title('Confusion matrix for Variant Effect Prediction (Initial Training)',  y=1.1)
    plt.ylabel("True label")
    plt.xlabel("Predicted Label")
    plt.subplots_adjust(left=0.1, top=0.8, bottom = 0.1)
    plt.savefig("output_plots/confusion_matrix_plot-initial_train.jpg", transparent = True)
    plt.show()

    # classification report
    print("\nClassification Report (Initial Training):\n", 
            classification_report(
                y_test, y_pred, 
                target_names=["Benign & Likely-Benign: 0", 
                            "Pathogenic, Likely-Pathogenic, Uncertain-Significance (VUS): 1"
                ]
            )
    )

    # plot ROC curve (Receiver Operating Characteristic Curve)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score[::,1])
    auc_score = metrics.roc_auc_score(y_test, y_score[::,1])
    print("ROC-AUC score (Initial Training):", auc_score)

    # plotting AUC curve
    # https://www.geeksforgeeks.org/machine-learning/ml-logistic-regression-using-python/
    plt.plot([0, 1], [0, 1], color='black', lw=3, linestyle='--')
    plt.plot(fpr, tpr, color='blue', lw=4, label=f'ROC Curve (AUC score = {auc_score:.2f})') 
    plt.title(f'ROC curve with an Accuracy of {accuracy:.2%} (Initial Training)') 
    plt.xlabel('FPR') 
    plt.ylabel('TPR') 
    plt.legend(loc="lower right")
    plt.savefig("output_plots/ROC curve-initial_train.jpg", transparent = True)
    plt.show()

    return df_MC1R, model

def cleaning_XY(X,Y):
    #Convert the nucleotides in ref and alt to one-hot encoding representing nucleotide A,C,G,T
    '''
    https://www.tandfonline.com/doi/full/10.1080/03610926.2021.1939382
    https://prod-edxapp.edx-cdn.org/assets/courseware/v1/f3989f5ebd854e332ec0b5c9890f7f41/c4x/BerkeleyX/CS190.1x/asset/CS190.1x_week4b.pdf
    https://www.datacamp.com/tutorial/one-hot-encoding-python-tutorial
    '''
    # one-hot encoding for ref column
    encode_ref = OneHotEncoder(handle_unknown='ignore', 
                               categories=[["A","C","G","T"]], # specify the order to match ref_encoded
                               sparse_output=True)
    # fit & transform the ref column into one-hot encoding with 4 columns each represent nucleotide
    ref_encoded = encode_ref.fit_transform(X[['ref']]).toarray() 
    # fit 'ref' instead of the nucleotides themselves to maintain features, prevent error
    X[['ref_A','ref_C','ref_G','ref_T']] = ref_encoded

    # one-hot encoding for alt column
    encode_alt = OneHotEncoder(handle_unknown='ignore', 
                               categories=[["A","C","G","T"]], # specify the order to match alt_encoded
                               sparse_output=True)
    # fit & transform the alt column into one-hot encoding with 4 columns each represent nucleotide
    alt_encoded = encode_alt.fit_transform(X[['alt']]).toarray() 
    # fit 'alt' instead of the nucleotides themselves to maintain features, prevent error
    X[['alt_A','alt_C','alt_G','alt_T']] = alt_encoded


    # drop the ref and alt columns in df X
    X = X.drop(['ref', 'alt'], axis = 1) # use axis = 1 for columns
    # print("X is:\n", X)

    # make sure X and Y df are floats
    X = X.apply(pd.to_numeric)
    Y = Y.apply(pd.to_numeric)
    # check data types
    # print(X.dtypes)

    return X,Y

def post_train_testing(df_MC1R, model):

    print('\n\n---------- Using MC1R on Built Model for Independent Testing ----------')

    # Selecting the features (split dataset features & target variables) for MC1R for testing again
    feature_cols = ['distance_to_tss', 'is_promoter',
           'is_enhancer', 'ref', 'alt', 'H3K36me3', 'H3K79me2', 'H3K27me3',
           'H3K9me3', 'E2F4', 'MYC', 'EZH2', 'CBX8', 'SUZ12', 'ESR1', 'TCF12',
           'FOSL1', 'GABPA', 'GC_content_up', 'GC_content_down']
    # drop(columns=['gene', 'chr', 'pos', 'rs', 'upstream_seq', 'downstream_seq','label'])
    X = df_MC1R[feature_cols]
    Y = df_MC1R["label_0-1"]

    # ref and alt to one-hot encoding representing nucleotide A,C,G,T
    X,Y = cleaning_XY(X,Y) 

    # before merging, make sure epigenetic features are float
    df_MC1R = df_MC1R.astype({'H3K36me3':float, 'H3K79me2':float, 
                                      'H3K27me3':float, 'H3K9me3':float, 
                                      'E2F4':float, 'MYC':float, 'EZH2':float, 
                                      'CBX8':float, 'SUZ12':float, 'ESR1':float, 
                                      'TCF12':float,'FOSL1':float, 'GABPA':float})

    # merge test dataset with original MC1R dataset to see which intron snv
    X_test_merge = pd.merge(X, df_MC1R, 
            on = None, #['distance_to_tss', 'is_promoter','is_enhancer'],
            how = "left")
    X_test_merge = X_test_merge[['gene', 'rs', 'chr', 'pos', 'ref', 
                                'alt', 'label_0-1']]

    # print("Given MC1R test dataset to model merged with true labels for reference:\n", X_test_merge)
    # print("\nTrue labels for test dataset:\n", y_test.to_numpy()) # true labels or use .values

    # predicted labels from X_test inputted into model: pathogenic=1 or benign=0
    y_pred = model.predict(X)
    # print("Predicted labels for MC1R test dataset:\n", y_pred) # predicted labels


    # Present Finalized df comparing predictions and true labels
    X_test_merge["pred_label"] = y_pred # add predicted labels to table

    # predict the probabilities/likelihood of these predictions
    y_score = model.predict_proba(X)
    # print(f"""\nScores for each predicted label ({model.classes_}) 
    # in each row of dataset:\n""", y_score) # likelihood of labels
    # add these scores to X_test_merge df
    X_test_merge["Likelihood_Pathogenic/Uncertain Sig"] = y_score[:,1]

    X_test_merge.rename(columns={"label_0-1":"true_label", "rs": "Intron_SNV"}, inplace=True) # change colname for true labels
    print("""\nComparing true labels with predicted labels, 
    including the predicted likelihood of pathogenicity/uncertain significance (post-train MC1R):\n""", X_test_merge)


    '''
    3. Evaluating the Model - Performance Analysis
    https://www.geeksforgeeks.org/machine-learning/ml-logistic-regression-using-python/
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    '''
    # find the accuracy of the model
    accuracy = accuracy_score(Y, y_pred) # (y_true, y_pred)
    # print("Accuracy: {:.2f}%".format(accuracy*100))
    print(f"\nAccuracy (post-train MC1R): {accuracy:.2%}")

    # confusion matrix
    # ith row and jth column (TN 0,0; FN 1,0; TP 1,1; FP 0,1)
    cf_matrix = metrics.confusion_matrix(Y, y_pred)
    print("Confusion Matrix (post-train MC1R):\n", cf_matrix)

    # plot confusion matrix: "Visualizing confusion matrix using a heatmap"
    # https://www.datacamp.com/tutorial/understanding-logistic-regression-python
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = ["Benign", "Pathogenic/Uncertain"]
    fig, ax = plt.subplots()
    num_ticks = np.arange(len(labels))
    plt.xticks(num_ticks, labels)
    plt.yticks(num_ticks, labels)

    # generate the heatmap
    sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap="coolwarm", fmt="g")
    ax.xaxis.set_label_position("bottom")
    plt.tight_layout()
    plt.title('Confusion matrix for Variant Effect Prediction (post-train MC1R)',  y=1.1)
    plt.ylabel("True label")
    plt.xlabel("Predicted Label")
    plt.subplots_adjust(left=0.1, top=0.8, bottom = 0.1)
    plt.savefig("output_plots/confusion_matrix_plot-MC1R.jpg", transparent = True)
    plt.show()

    # classification report
    print("\nClassification Report (post-train MC1R):\n", 
            classification_report(
                Y, y_pred, 
                target_names=["Benign & Likely-Benign: 0", 
                              "Pathogenic, Likely-Pathogenic, Uncertain-Significance (VUS): 1"
                ]
            )
    )

    # plot ROC curve (Receiver Operating Characteristic Curve)
    fpr, tpr, thresholds = metrics.roc_curve(Y, y_score[::,1])
    auc_score = metrics.roc_auc_score(Y, y_score[::,1])
    print("\nROC-AUC score (post-train MC1R):", auc_score)

    # plotting AUC curve
    # https://www.geeksforgeeks.org/machine-learning/ml-logistic-regression-using-python/
    plt.plot([0, 1], [0, 1], color='black', lw=3, linestyle='--')
    plt.plot(fpr, tpr, color='blue', lw=4, label=f'ROC Curve (AUC score = {auc_score:.2f})') 
    plt.title(f'ROC curve with an Accuracy of {accuracy:.2%} (post-train MC1R)',fontsize=11) 
    plt.xlabel('FPR') 
    plt.ylabel('TPR') 
    plt.legend(loc="lower right", fontsize=8.5)
    plt.savefig("output_plots/ROC curve-MC1R.jpg", transparent = True)
    plt.show()


if __name__ == "__main__":
    os.makedirs("output_plots", exist_ok=True) # create a directory to save the plots
    df_MC1R, model = initial_training() # clean data and train the model with the testing data (not MC1R)
    post_train_testing(df_MC1R, model) # test MC1R on the trained model