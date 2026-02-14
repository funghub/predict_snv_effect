# predict_snv_effect
This project utilizes a logistic regression, a supervised machine learning model, to predict the pathogenicity of specific non-coding single nucleotide variants in melanoma given epigenetic features.
Pandas was used to organize, clean, and merge the data. Scikit-learn was used to train and evaluate the model, along with one-hot encoding for nucleotides because logistic regression only takes numerical features.
Pyplot from matplotlib and seaborn was used to generate a heatmap of the confusion matrix and plot the ROC-AUC curve.

In the data set, the machine learning model needed to predict one of the two labels:
- Pathogenic, Likely-Pathogenic, Uncertain-Significance (VUS) = 1
- Benign and Likely-Benign = 0

## Data Source
Multimodal dataset is from NCBI variant viewer and Enricher to source histone midifications, transcription factors, and the local DNA sequence.
- 30 non coding single nucleotide polymorphisms (SNPs) were used from NCBI variant viwer
- histone modifications of those variants were found on Enricher for transcription factor binding score from ENCODE TF ChIP-seq 2015 and histone modifications from ENCODE Histone Modifications 2015.


Additional SNP was used to evaluate the model independently after training. MC1R was used as a test case.

## Results (additional results found in final_code_results.txt and output_plots directory)
### Initial Training
Accuracy (Initial Training): 62.50%
ROC-AUC score (Initial Training): 0.6

### Post Training Independent Testing with MC1R
Accuracy (post-train MC1R): 33.33%
ROC-AUC score (post-train MC1R): 0.5

These results indicatte that more diverse datasets need to be used for training and additional data is necessary to improve ROC-AUC score and accuracy.
