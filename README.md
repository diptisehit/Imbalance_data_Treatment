![image](https://github.com/user-attachments/assets/70e87c6e-e1af-4287-984a-ea0daa9b9e35)
# What is Imbalanced Data ?

Imbalanced data pertains to datasets where the distribution of observations in the target class is uneven. In other words, one class label has a significantly higher number of observations, while the other has a notably lower count.

Algorithms may get biased towards the majority class and thus tend to predict output as the majority class. Minority class observations look like noise to the model and are ignored by the model. Imbalanced dataset gives misleading accuracy score.

If minority class*2>majority class, then data is balanced, otherwise the data is imbalanced.

# How to handle it ?

1. Resampling (Undersampling and Oversampling) :  
   
   
   - Under Sampling : It consists of removing samples from the majority class which can cause a loss of information.
     
        from imblearn.under_sampling import RandomUnderSampler
     
        under = RandomUnderSampler()
     
        x_under, y_under = under.fit_resample(x,y)
  
     **class_0_under = class_0.sample(class_count_1)**

     
   - Over Sampling : It consist of adding more examples from the minority class which can cause overfishing.
  
        from imblearn.over_sampling import RandomOverSampler
  
     
        over = RandomOverSampler()
  
     
        x_over, y_over = over.fit_resample(x,y)
  
     **class_1_over = class_1.sample(class_count_0, replace=True)**
  
2. Synthetic Minority Oversampling Technique (SMOTE):

   This technique generates synthetic data for the minority class. It works by randomly picking a point from the minority class and computing the k-nearest neighbors for this point. The synthetic points are added between the chosen point and its neighbors.

    from imblearn.over_sampling import SMOTE

    smote = SMOTE()

    x_smote, y_smote = smote.fit_resample(x, y)

3. NearMiss :NearMiss is an under-sampling technique.

    from imblearn.under_sampling import NearMiss

    nm = NearMiss()

    x_nm, y_nm = nm.fit_resample(x, y)

4.  Change the Performance Metric :Accuracy is not the best metric to use when evaluating imbalanced datasets, as it can be misleading. Metrics that can provide better insight are:

    - Confusion Matrix: a table showing correct predictions and types of incorrect predictions.
    - Precision: the number of true positives divided by all positive predictions. Precision is also called Positive Predictive Value. It is a measure of a classifier’s exactness. Low precision indicates a high number of false positives.
    - Recall: the number of true positives divided by the number of positive values in the test data. The recall is also called Sensitivity or the True Positive Rate. It is a measure of a classifier’s completeness. Low recall indicates a high number of false negatives.
    - F1: Score: the weighted average of precision and recall.
    - Area Under ROC Curve (AUROC): AUROC represents the likelihood of your model distinguishing observations from two classes.
5. Use Penalize Algorithms (Cost-Sensitive Training) :During training, we can use the argument class_weight=’balanced’ to penalize mistakes on the minority class by an amount proportional to how under-represented it is.

        from sklearn.svm import SVC

        svc_model = SVC(class_weight='balanced', probability=True)

        svc_model.fit(x_train, y_train)

6. Change the Algorithm : Tree base algorithm work by learning a hierarchy of if/else questions. This can force both classes to be addressed. e.g.Decision trees, Random Forests, Gradient Boosted Trees, etc. AdaBoost and XGBoost are known for their ability to handle imbalanced datasets effectively. 
   
