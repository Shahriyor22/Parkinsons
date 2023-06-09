import streamlit as st
import numpy as np
import pandas as pd
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

st.set_page_config(page_title='Parkinson\'s Disease Detection',                   
                   page_icon='brain',
                   layout='wide',
                   initial_sidebar_state='auto')

loaded_model = pickle.load(open('D:/Shaxriyor/Parkinsons/trained_model.sav', 'rb'))
dataset = pd.read_csv('D:/Shaxriyor/Parkinsons/parkinsons.csv')
scaler = StandardScaler()
dataset_scaled = scaler.fit_transform(dataset.drop(['name','status'], axis=1))
score = loaded_model.score(dataset_scaled, dataset['status'])

with st.sidebar:
    selected = option_menu('Navigation',
                           ['Home', 'Detection', 'Data Info', 'Visualization'],
                           icons=['house', 'person', 'info', 'activity'],
                           menu_icon="cast",
                           default_index=0)                                          

def parkinsons_detection(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = loaded_model.predict(std_data)
    print(prediction)
    if (prediction[0] == 0):
      return 0
    else:
      return 1   
 
def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(f'''
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        ''', unsafe_allow_html=True)

# Home Page
if (selected == "Home"):
      
    st.title('Parkinson\'s Disease Detection')
    add_bg_from_local('D:/Shaxriyor/Parkinsons/bg.jpg')
    
    image = Image.open('D:/Shaxriyor/Parkinsons/parkinsons.jpg')
    st.image(image)
    
    st.markdown('''
                <p style="font-size:20px;">
                Parkinson's disease is a chronic condition that affects both the neurological system and the bodily components that are under the control of the nervous system. Symptoms emerge gradually. The initial sign could be a slight tremor in just one hand. Although tremors are typical, the disease might also make you stiff or move more slowly.
                This web application will assist you in determining whether a person has Parkinson's disease or not. This app employs the K-Nearest Neighbors Classifier to analyze the values of various feature variables in order to identify the causes of similar problematic occurrences in the future.
                </p>
                ''', unsafe_allow_html=True)

# Detection Page
if (selected=="Detection"):
    
    st.title('Detection Page')
    
    st.markdown('''<p style="font-size:24px">This app uses <b style="color:green">K-Nearest Neighbors Classifier</b> for the detection of Parkinson's disease.</p>''', unsafe_allow_html=True)    
    
    with st.expander("View attribute details"):
        st.warning('''
                   MDVPꓽFo(Hz) – average vocal fundamental frequency  
                   MDVPꓽFhi(Hz) – maximum vocal fundamental frequency  
                   MDVPꓽFlo(Hz) – minimum vocal fundamental frequency  
                   MDVPꓽJitter(%), MDVPꓽJitter(Abs), MDVPꓽRAP, MDVPꓽPQ, JitterꓽDDP – several measures of variation in fundamental frequency  
                   MDVPꓽShimmer, MDVPꓽShimmer(dB), ShimmerꓽAPQ3, ShimmerꓽAPQ5, MDVPꓽAPQ, ShimmerꓽDDA – several measures of variation in amplitude  
                   NHR, HNR – two measures of ratio of noise to tonal components in the voice  
                   RPDE, D2 – two nonlinear dynamical complexity measures  
                   DFA – signal fractal scaling exponent  
                   spread1, spread2, PPE – three nonlinear measures of fundamental frequency variation
                   ''')
    
    st.subheader("Select Values:") 
    
    MDVP_f0 = st.slider('Average Vocal Fundamental Frequency', int(dataset["MDVP:Fo(Hz)"].min()), int(dataset["MDVP:Fo(Hz)"].max()))
    MDVP_fhi = st.slider('Maximum Vocal Fundamental Frequency', int(dataset["MDVP:Fhi(Hz)"].min()), int(dataset["MDVP:Fhi(Hz)"].max()))
    MDVP_flo = st.slider('Minimum Vocal Fundamental Frequency', int(dataset["MDVP:Flo(Hz)"].min()), int(dataset["MDVP:Flo(Hz)"].max()))
    MDVP_jitter = st.slider('MDVP Jitter in Percentage', float(dataset["MDVP:Jitter(%)"].min()), float(dataset["MDVP:Jitter(%)"].max()))
    MDVP_jitter_abs = st.slider('MDVP Absolute Jitter in ms', float(min(dataset["MDVP:Jitter(Abs)"])), float(max(dataset["MDVP:Jitter(Abs)"])))
    MDVP_rap = st.slider('MDVP Relative Amplitude Perturbation', float(dataset["MDVP:RAP"].min()), float(dataset["MDVP:RAP"].max()))
    MDVP_ppq = st.slider('MDVP Five-Point Period Perturbation Quotient', float(dataset["MDVP:PPQ"].min()), float(dataset["MDVP:PPQ"].max()))
    jitter_ddp = st.slider('Average Absolute Difference of Differences Between Jitter Cycles', float(dataset["Jitter:DDP"].min()), float(dataset["Jitter:DDP"].max()))
    MDVP_shimmer = st.slider('MDVP Local Shimmer', float(dataset["MDVP:Shimmer"].min()), float(dataset["MDVP:Shimmer"].max()))
    MDVP_shimmer_dB = st.slider('MDVP Local Shimmer in dB', float(dataset["MDVP:Shimmer(dB)"].min()), float(dataset["MDVP:Shimmer(dB)"].max()))
    shimmer_apq3 = st.slider('Three-Point Amplitude Perturbation Quotient', float(dataset["Shimmer:APQ3"].min()), float(dataset["Shimmer:APQ3"].max()))
    shimmer_apq5 = st.slider('Five-Point Amplitude Perturbation Quotient', float(dataset["Shimmer:APQ5"].min()), float(dataset["Shimmer:APQ5"].max()))
    MDVP_apq11 = st.slider('MDVP 11-Point Amplitude Perturbation Quotient', float(dataset["MDVP:APQ"].min()), float(dataset["MDVP:APQ"].max()))
    shimmer_dda = st.slider('Average Absolute Differences Between the Amplitudes of Consecutive Periods', float(dataset["Shimmer:DDA"].min()), float(dataset["Shimmer:DDA"].max()))
    NHR = st.slider('Noise-to-Harmonics Ratio', float(dataset["NHR"].min()), float(dataset["NHR"].max()))
    HNR = st.slider('Harmonics-to-Noise Ratio', float(dataset["HNR"].min()), float(dataset["HNR"].max()))
    RPDE = st.slider('Recurrence Period Density Entropy Measure', float(dataset["RPDE"].min()), float(dataset["RPDE"].max()))
    DFA = st.slider('Signal Fractal Scaling Exponent of Detrended Fluctuation Analysis', float(dataset["DFA"].min()), float(dataset["DFA"].max()))
    spread1 = st.slider('Two Nonlinear Measures of Fundamental', float(dataset["spread1"].min()), float(dataset["spread1"].max()))
    spread2 = st.slider('Frequency Variation', float(dataset["spread2"].min()), float(dataset["spread2"].max()))
    D2 = st.slider('Correlation Dimension', float(dataset["D2"].min()), float(dataset["D2"].max()))
    PPE = st.slider('Pitch Period Entropy', float(dataset["PPE"].min()), float(dataset["PPE"].max())) 
    
    if st.button('Parkinson\'s Disease Test Result'):
        diagnosis = parkinsons_detection([MDVP_f0, MDVP_fhi, MDVP_flo, MDVP_jitter, MDVP_jitter_abs, MDVP_rap, MDVP_ppq, jitter_ddp, MDVP_shimmer, MDVP_shimmer_dB, shimmer_apq3, shimmer_apq5, MDVP_apq11, shimmer_dda, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE])
        st.info("Diagnosed successfully!")
        if(diagnosis == 0):
            st.success("The person does not have Parkinson\'s disease.")
        else:
            st.error("The person has Parkinson\'s disease.")
        st.write("The model used can be trusted by doctors and has an accuracy of ", (score*100),"%.")
                
# Data Info Page
if (selected == "Data Info"):
    
    st.title('Data Info Page')
    
    st.subheader("View Data")
    
    with st.expander("View data"):
        st.dataframe(dataset)
        
    st.subheader("Columns Description:")
    
    if st.checkbox("View Summary"):
        st.dataframe(dataset.describe())
        
    col_name, col_dtype, col_data = st.columns(3)
    
    with col_name:
        if st.checkbox("Column Names"):
            st.dataframe(dataset.columns)
            
    with col_dtype:
        if st.checkbox("Columns data types"):
            dtypes = dataset.dtypes.apply(lambda x: x.name)            
            st.dataframe(dtypes)
            
    with col_data: 
        if st.checkbox("Columns Data"):
            col = st.selectbox("Column Name", list(dataset.columns))
            st.dataframe(dataset[col])
            
    st.markdown('''<p style="font-size:24px"><a href="https://raw.githubusercontent.com/chaitanyabaranwal/ParkinsonAnalysis/master/parkinsons.csv" target=_blank style="text-decoration:none;">Get Dataset</a></p>''', unsafe_allow_html=True)

# Visualization Page
if (selected == "Visualization"):
    
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.title("Visualize the Parkinson's disease detection")
    
    dataset.rename(columns = {"MDVP:Fo(Hz)": "AVFF",}, inplace = True)
    dataset.rename(columns = {"MDVP:Fhi(Hz)": "MAVFF",}, inplace = True)
    dataset.rename(columns = {"MDVP:Flo(Hz)": "MIVFF",}, inplace = True)
    
    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        fig = plt.figure(figsize = (10, 6))
        ax = sns.heatmap(dataset.iloc[:, 1:].corr(), annot=True)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom+0.5, top-0.5)
        st.pyplot(fig)

    if st.checkbox("Show Scatter Plot"):
        st.subheader("Scatter Plot")
        figure, axis = plt.subplots(2, 2,figsize=(15,10))
        sns.scatterplot(ax=axis[0, 0], data=dataset, x='AVFF', y='MAVFF', hue='status')
        axis[0, 0].set_title("Oversampling Minority Scatter")
        sns.countplot(ax=axis[0, 1], x="status", data=dataset)
        axis[0, 1].set_title("Oversampling Minority Count")  
        sns.scatterplot(ax=axis[1, 0], data=dataset, x='AVFF', y='MAVFF', hue='status')
        axis[1, 0].set_title("Undersampling Majority Scatter")  
        sns.countplot(ax=axis[1, 1], x="status", data=dataset)
        axis[1, 1].set_title("Undersampling Majority Count")
        st.pyplot()

    if st.checkbox("Display Boxplot"):
        st.subheader("Boxplot")
        fig, ax = plt.subplots(figsize=(15,5))
        dataset.boxplot(['AVFF', 'MAVFF', 'MIVFF', 'HNR'], ax=ax)
        st.pyplot()

    if st.checkbox("Show Sample Results"):
        st.subheader("Sample Results")
        safe = (dataset['status']==0).sum()
        prone = (dataset['status']==1).sum()
        data = [safe, prone]
        labels = ['Safe', 'Prone']
        colors = sns.color_palette('pastel')[0:7]
        plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
        st.pyplot()

    if st.checkbox("Plot Confusion Matrix"):
        st.subheader("Confusion Matrix")        
        X_train, X_test, Y_train, Y_test = train_test_split(dataset.drop(['name','status'], axis=1), dataset['status'], test_size=0.25, random_state=2)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        model = KNeighborsClassifier(n_neighbors=10)
        model.fit(X_train, Y_train)
        plt.figure(figsize=(10, 6))
        ConfusionMatrixDisplay.from_estimator(model, dataset_scaled, dataset['status'],  values_format='d')
        st.pyplot()