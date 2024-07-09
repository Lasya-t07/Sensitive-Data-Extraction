# Sensitive-Data-Extraction
In today's world personal details such as name, phone number, email-id and password are the most crucial elements to protect. In healthcare domain information, such as personal details, details of diseases, and details of treatments, is extremely sensitive.
Protecting sensitive information is therefore a fundamental and top need in the medical industry. 
The entire attribute encryption take more processing time and reduce data usability to authorized users. 
To provide security to personal data and reduce processing time, the need for identifying and securing sensitive information has become more crucial than ever. 

The entire attribute protection slows down the processing of the data and decreases its usability for authorized users. Applying protection to a sensitive attribute instead of an entire attribute increases usability for authorized users and protects sensitive attributes. 

To classify sensitive attributes from a structured and semi-structured data by using fuzzy-rules and natural language processing techniques and also ML and DL techniques. Applying protection to sensitive attributes by using attribute-based encryption.    

CNN
1.Model Creation: A sequential model is created using Keras. The model includes an Embedding layer, a Batch Normalization layer, a 1D Convolution Layer, a Dropout layer, a Global Average Pooling layer, another Batch Normalization layer, a Dense layer with ReLU activation, another Dropout layer a final Dense layer with sigmoid activation.
2.Compile Model: The model is compiled with cross-entropy as the loss function, Adam as the optimizer.
3.Display of model summary: Model summary is displayed.
4.Set the Epochs: The number of epochs for training is set to 10.
5. Model is trained using fit method.
6.Passing validation data to fit method, model evaluates after each epoch.
7.The model is evaluated by accuracy measure.

LSTM
The model layers include an Embedding layer, two LSTM layers, a Dropout layer, and a Dense layer. The Embedding layer transforms each text input into a dense vector of fixed size. The LSTM layers process the sequence data, capturing long-term dependencies in the text. 
Forward pass  passes the input through each layer in turn and returns the output of the Dense layer.
Error handling is to ensure that the input to the model has the correct shape. If the input does not have rank 2 or if the second dimension of the input is not equal to maxlen an error is raised.
There were several pre-processing techniques like tokenizer and conversion to sequences along with embeddings like Word2Vec, GloVe.

BERT
BERT, which stands for Bidirectional Encoder Representations from Transformers, is a state-of-the-art machine learning model for natural language processing tasks. 
BERT is a two-step process model. First, it is pre-trained on a large corpus of text data in an unsupervised manner. This pre-training step allows BERT to learn the general language understanding capability. Then, it is fine-tuned on a specific task (like text classification, sentiment analysis, question answering, etc.) with a smaller amount of task-specific data.
Unlike some other models that read the text input either from left to right or from right to left, BERT reads the entire sequence of words at once.
BERT is based on the Transformer architecture, which uses self-attention mechanisms and is capable of handling long-range dependencies in text.

Random Forest and Decision Tree
1.The data is processed using CountVectorizer which converts collection of text documents to a matrix of token counts.
2. Now a RandomForestClassifier is created with n_estimators=1 and the model is fitted by X_train and y_train and then the model is tested on X_test and accuracy is observed.
3.Similarly DecisionTreeClassifier is created with criterion=‘entropy’, random_state=0,max_depth=3 and model is trained, tested and evaluated.

SVM
1.The data is processed using CountVectorizer which converts collection of text documents to a matrix of token counts.
2.SVM   with different kernel functions is implemented to find out the best suited kernel.

Fuzzy
1.Define universe discourse for  'username', 'password', 'email', and 'phone_number’ as inputs, and 'sensitivity’ as the output.
2. Membership functions are defined with automf(3) function which automatically creates three fuzzy sets for each input variable: poor, average, and good. For the output variable 'sensitivity', three fuzzy sets are manually created: low, medium, and high with trimf().
3. Define Fuzzy rules: There are seven rules defined here. For example,if the username, password, email, and phone number are all 'poor', then the sensitivity is 'low’. 
4.Define Control System : This system takes the inputs and applies the rules to determine the output. 
5. Simulation: Create a control system simulation to apply fuzzy control. This will be used later to input the values and get the output. Regular expressions are used to identify the presence of sensitive attributes and then they are valued based on their presence.
6.Sensitivity graph: Use view() function to observe how sensitivity varies

Data will be provided on need.
