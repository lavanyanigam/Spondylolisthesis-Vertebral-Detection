import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
 
st.set_page_config(page_title="Lumbar Spine Spondylolisthesis Classifier", layout="wide")

levels = ['L1a', 'L1b', 'L2a', 'L2b', 'L3a', 'L3b', 'L4a', 'L4b', 'L5a', 'L5b', 'S1a']
POINT_SEQUENCE = []
for lvl in levels:
    POINT_SEQUENCE.append(f"{lvl}_1")  # anterior corner
    POINT_SEQUENCE.append(f"{lvl}_2")  # posterior corner
 
TOTAL_POINTS = len(POINT_SEQUENCE)

def euclidean_distance(p1x, p1y, p2x, p2y):
        return np.sqrt((p1x-p2x)**2+(p1y-p2y)**2)

def angle_calc(p1x, p1y, p2x, p2y):
    return np.degrees(np.arctan2((p1y-p2y),(p1x-p2x)))

def angle_calc_lines(p1x, p1y, p2x, p2y,p3x, p3y, p4x, p4y):
    m1=(p1y-p2y)/(p1x-p2x)
    m2=(p3y-p4y)/(p3x-p4x)
    return np.degrees(np.arctan2((m1-m2),(1+m1*m2)))
    
def cobbs_angle(max1,min1):
    return (np.degrees(np.pi)-max1+min1)

def extract_geometric_features(df):
    features=pd.DataFrame(index=df.index)
    features['gender']=df['gender']
    features['age'] = df['age']
    if 'type' in df.columns:
        features['type'] = df['type']
    #Vertebral Tilt
    levels = ['L1a', 'L1b', 'L2a', 'L2b', 'L3a', 'L3b', 'L4a', 'L4b', 'L5a', 'L5b', 'S1a']
    for lvl in levels:
        features[f'v_tilt_{lvl}'] = angle_calc(
            df[f'{lvl}_1c'], df[f'{lvl}_1r'], 
            df[f'{lvl}_2c'], df[f'{lvl}_2r']
            )
    #Intersegmental Angles
    features['int_seg_l1_l2']=angle_calc_lines(df['L1b_1c'], df['L1b_1r'], df['L1b_2c'], df['L1b_2r'],df['L2a_1c'], df['L2a_1r'], df['L2a_2c'], df['L2a_2r'])
    features['int_seg_l2_l3']=angle_calc_lines(df['L2b_1c'], df['L2b_1r'], df['L2b_2c'], df['L2b_2r'],df['L3a_1c'], df['L3a_1r'], df['L3a_2c'], df['L3a_2r'])
    features['int_seg_l3_l4']=angle_calc_lines(df['L3b_1c'], df['L3b_1r'], df['L3b_2c'], df['L3b_2r'],df['L4a_1c'], df['L4a_1r'], df['L4a_2c'], df['L4a_2r'])
    features['int_seg_l4_l5']=angle_calc_lines(df['L4b_1c'], df['L4b_1r'], df['L4b_2c'], df['L4b_2r'],df['L5a_1c'], df['L5a_1r'], df['L5a_2c'], df['L5a_2r'])
    features['int_seg_l5_s1']=angle_calc_lines(df['L5b_1c'], df['L5b_1r'], df['L5b_2c'], df['L5b_2r'],df['S1a_1c'], df['S1a_1r'], df['S1a_2c'], df['S1a_2r'])
    
    #cobb's angle
    
    tilt_cols = [f'v_tilt_{lvl}' for lvl in levels]
    features['cobbs_angle'] = features[tilt_cols].max(axis=1) - features[tilt_cols].min(axis=1)
            
    #Anterior and Posterior Disc Height
    
    for i in range(1,len(levels),2):
        lvl =levels[i]
        if lvl.endswith('b'):
            features[f'a_disl_h_{lvl}_{levels[i+1]}'] = euclidean_distance(
                    df[f'{lvl}_1c'], df[f'{lvl}_1r'], 
                    df[f'{levels[i+1]}_1c'], df[f'{levels[i+1]}_1r']
                )
            
            features[f'p_disl_h_{lvl}_{levels[i+1]}'] = euclidean_distance( 
                    df[f'{lvl}_2c'], df[f'{lvl}_2r'], 
                    df[f'{levels[i+1]}_2c'], df[f'{levels[i+1]}_2r']
                )  
    
            features[f'compr_ratio_{lvl}_{levels[i+1]}']= features[f'a_disl_h_{lvl}_{levels[i+1]}'] /( features[f'p_disl_h_{lvl}_{levels[i+1]}'] + 1e-9)
    
    #Aspect Ratio , Slippage Distance between L4 and L5
    
    #Aspect Ratio
    
    for i in range(0, len(levels)-1, 2):
        lvl_a = levels[i]       # Superior endplate
        lvl_b = levels[i+1]     # Inferior endplate 
        
        if lvl_a.endswith('a') and lvl_b.endswith('b'):
            
            ant_vert_h = euclidean_distance(
                df[f'{lvl_a}_1c'], df[f'{lvl_a}_1r'], 
                df[f'{lvl_b}_1c'], df[f'{lvl_b}_1r']
            )
            
            post_vert_h = euclidean_distance(
                df[f'{lvl_a}_2c'], df[f'{lvl_a}_2r'], 
                df[f'{lvl_b}_2c'], df[f'{lvl_b}_2r']
            )
            
            endplate_width = euclidean_distance(
                df[f'{lvl_a}_1c'], df[f'{lvl_a}_1r'], 
                df[f'{lvl_a}_2c'], df[f'{lvl_a}_2r']
            )
            
            avg_vert_h = (ant_vert_h + post_vert_h) / 2.0
            vert_name = lvl_a[:2] 
            
            features[f'aspect_ratio_{vert_name}'] = endplate_width / (avg_vert_h + 1e-9)

    # Slippage Distance between L3, L4, L5 and S1 with sign
    
    features[f'slip_dist_L3_L4'] = df['L3b_2c'] - df['L4a_2c']
    features[f'slip_dist_L4_L5'] = df['L4b_2c'] - df['L5a_2c']
    features[f'slip_dist_L5_S1'] = df['L5b_2c'] - df['S1a_2c']
    
    #Tangential vector projection
    
    v_L5_x = df['L5a_2c'] - df['S1a_2c']
    v_L5_y = df['L5a_2r'] - df['S1a_2r']
    v_slip_x = df['L4b_2c'] - df['L5a_2c']
    v_slip_y = df['L4b_2r'] - df['L5a_2r']
    
    # Cross product
    features['L4_L5_step_off'] =  (v_slip_x * v_L5_y - v_slip_y * v_L5_x) / (np.sqrt(v_L5_x**2 + v_L5_y**2)+ 1e-9)
    
    #Taillard’s Method for Percentage Spondylolisthesis
    lvl='L4b'
    features[f'perct_spond']= 100*features[f'slip_dist_L4_L5']  / (euclidean_distance( df[f'{lvl}_1c'], df[f'{lvl}_1r'], df[f'{lvl}_2c'], df[f'{lvl}_2r']) + 1e-9)


    return features


@st.cache_resource 
def load_model():
    model=xgb.XGBClassifier()
    model.load_model('model.json')
    return model

CLASS_NAMES = {0: "Normal", 1: "Anterolisthesis", 2: "Retrolisthesis"}

st.title("Lumbar Spine Spondylolisthesis Classifier")
st.subheader("Upload a lateral X-ray, click each landmark in order from L1 to S1, anterior to posterior, and get a prediction with explanation.")
st.divider()

if "points" not in st.session_state:
    st.session_state.points = {}

if "last_click" not in st.session_state:
    st.session_state.last_click = None


col1, col2 = st.columns([2, 1])
 
with col1:
    xray_file = st.file_uploader("Upload lateral Lumbar spine X-Ray", type=["jpg","JPG","jpeg","png"])
    if xray_file:
        image = Image.open(xray_file).convert("RGB")

        
        MAX_WIDTH = 700
        scale = MAX_WIDTH / image.width if image.width > MAX_WIDTH else 1.0
        disp_w, disp_h = int(image.width * scale), int(image.height * scale)
        display_img = image.resize((disp_w, disp_h))
        draw = ImageDraw.Draw(display_img)

        
        for label, (x, y) in st.session_state.points.items():
            dx, dy = x * scale, y * scale
            r = 4
            draw.ellipse((dx - r, dy - r, dx + r, dy + r), fill="red")

        current_index = len(st.session_state.points)
        if current_index < TOTAL_POINTS:
            current_label = POINT_SEQUENCE[current_index]
            st.info(f"Click point **{current_index + 1} / {TOTAL_POINTS}**: **{current_label}**")
        else:
            st.success("All points collected.")

        coords = streamlit_image_coordinates(display_img, key="xray_click")

        if coords is not None and current_index < TOTAL_POINTS:
            raw_click = (coords["x"], coords["y"])

            
            if st.session_state.get("last_click") != raw_click:
                st.session_state.last_click = raw_click

                
                orig_x = raw_click[0] / scale
                orig_y = raw_click[1] / scale

                label = POINT_SEQUENCE[current_index]
                st.session_state.points[label] = (orig_x, orig_y)
                st.rerun()

        if st.button("Reset points"):
            st.session_state.points = {}
            st.session_state.last_click = None
            st.rerun()

with col2:
    st.subheader("Patient Info")
    age=st.number_input("Age", min_value=1,max_value=120,value=45)
    gender=st.radio("Gender",["Male","Female"])
    gender_val=0 if gender=="Male" else 1
    xray_view = st.selectbox("X-ray view", ["LA"])
    type_val = 1 if xray_view == "LA" else 0

    st.subheader("Points collected")
    st.write(f"{len(st.session_state.points)} / {TOTAL_POINTS}")
    if st.session_state.points:
        st.dataframe(pd.DataFrame(st.session_state.points, index=["x", "y"]).T)
 
    run_prediction = st.button(
        "Predict",
        disabled=len(st.session_state.points) < TOTAL_POINTS,
        type="primary"
    )

if run_prediction and len(st.session_state.points)==TOTAL_POINTS:
    row = {}
    for lvl in levels:
        for n in [1,2]:
            label=f"{lvl}_{n}"
            x, y = st.session_state.points[label]
            row[f"{lvl}_{n}c"] = x
            row[f"{lvl}_{n}r"] = y
    row['age'] = age
    row['gender'] = gender_val
    row['type'] = type_val
    raw_df = pd.DataFrame([row], index=["patient_1"])
    X = extract_geometric_features(raw_df)
    
    model=load_model()
    expected_cols = model.get_booster().feature_names
    X = X.reindex(columns=expected_cols, fill_value=0)
 
    pred_class = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
 
    st.divider()
    st.header(f"Prediction: {CLASS_NAMES.get(pred_class, pred_class)}")
    proba_df = pd.DataFrame({
        "Class": [CLASS_NAMES.get(i, i) for i in range(len(pred_proba))],
        "Probability": pred_proba
    })
    st.bar_chart(proba_df.set_index("Class"))
 
    st.subheader("Why did the model predict this? (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_exp = explainer(X)
 
    fig, ax = plt.subplots(figsize=(7, 4))

    if shap_exp.values.ndim == 3:
        shap.plots.waterfall(shap_exp[0, :, pred_class], show=False)
    else:
        shap.plots.waterfall(shap_exp[0], show=False)
    st.pyplot(fig)
 
    with st.expander("Show extracted features"):
        st.dataframe(X.T.rename(columns={0: "value"}))
 
 