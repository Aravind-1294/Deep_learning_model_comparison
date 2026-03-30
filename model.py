import streamlit as st
import torch
import torchvision.models as models
from thop import profile, clever_format
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta


st.set_page_config(
    page_title="Model Comparison & AWS Cost Estimator",
    page_icon="🤖",
    layout="wide"
)


st.title("Deep Learning Model Comparison & AWS Cost Estimator")
st.markdown("Compare PyTorch models and estimate AWS SageMaker training costs")


AWS_INSTANCES = {
    "ml.t3.medium (CPU only)": {"price": 0.050, "gpu": "None", "memory": "4 GB"},
    "ml.g4dn.xlarge (NVIDIA T4)": {"price": 0.736, "gpu": "T4 (16GB)", "memory": "16 GB"},
    "ml.g5.xlarge (NVIDIA A10G)": {"price": 1.015, "gpu": "A10G (24GB)", "memory": "16 GB"},
    "ml.p3.2xlarge (NVIDIA V100)": {"price": 3.825, "gpu": "V100 (16GB)", "memory": "61 GB"},
    "ml.g4dn.12xlarge (4x T4)": {"price": 4.89, "gpu": "4x T4 (64GB)", "memory": "192 GB"},
    "ml.p3.8xlarge (4x V100)": {"price": 14.688, "gpu": "4x V100 (64GB)", "memory": "244 GB"},
}


MODEL_CATALOG = {
    "VGG16": {"constructor": models.vgg16, "family": "VGG", "depth": 16},
    "VGG19": {"constructor": models.vgg19, "family": "VGG", "depth": 19},
    "ResNet18": {"constructor": models.resnet18, "family": "ResNet", "depth": 18},
    "ResNet34": {"constructor": models.resnet34, "family": "ResNet", "depth": 34},
    "ResNet50": {"constructor": models.resnet50, "family": "ResNet", "depth": 50},
    "ResNet101": {"constructor": models.resnet101, "family": "ResNet", "depth": 101},
    "ResNet152": {"constructor": models.resnet152, "family": "ResNet", "depth": 152},
    "MobileNetV2": {"constructor": models.mobilenet_v2, "family": "MobileNet", "depth": 53},
    "MobileNetV3-Small": {"constructor": models.mobilenet_v3_small, "family": "MobileNet", "depth": 14},
    "MobileNetV3-Large": {"constructor": models.mobilenet_v3_large, "family": "MobileNet", "depth": 16},
    "EfficientNet-B0": {"constructor": models.efficientnet_b0, "family": "EfficientNet", "depth": 18},
    "EfficientNet-B1": {"constructor": models.efficientnet_b1, "family": "EfficientNet", "depth": 18},
    "EfficientNet-B2": {"constructor": models.efficientnet_b2, "family": "EfficientNet", "depth": 18},
    "EfficientNet-B3": {"constructor": models.efficientnet_b3, "family": "EfficientNet", "depth": 18},
    "DenseNet121": {"constructor": models.densenet121, "family": "DenseNet", "depth": 121},
    "DenseNet169": {"constructor": models.densenet169, "family": "DenseNet", "depth": 169},
    "DenseNet201": {"constructor": models.densenet201, "family": "DenseNet", "depth": 201},
}

@st.cache_resource
def get_model_specs(model_name):

    model_info = MODEL_CATALOG[model_name]
    model = model_info["constructor"](pretrained=False)
    

    total_params = sum(p.numel() for p in model.parameters())
    

    dummy_input = torch.randn(1, 3, 224, 224)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    flops_str, params_str = clever_format([flops, total_params], "%.3f")
    

    size_mb = total_params * 4 / (1024**2)
    
    return {
        "name": model_name,
        "family": model_info["family"],
        "depth": model_info["depth"],
        "total_params": total_params,
        "params_str": params_str,
        "flops": flops,
        "flops_str": flops_str,
        "size_mb": size_mb
    }

def estimate_training_time(flops, instance_type):


    base_flops = 4.1e9
    base_time_v100 = 20
    

    instance_speed = {
        "ml.t3.medium (CPU only)": 0.05,
        "ml.g4dn.xlarge (NVIDIA T4)": 0.6,
        "ml.g5.xlarge (NVIDIA A10G)": 0.8,
        "ml.p3.2xlarge (NVIDIA V100)": 1.0,
        "ml.g4dn.12xlarge (4x T4)": 2.4,
        "ml.p3.8xlarge (4x V100)": 4.0,
    }
    

    complexity_ratio = flops / base_flops
    speed_multiplier = instance_speed.get(instance_type, 1.0)
    

    estimated_hours = (base_time_v100 * complexity_ratio) / speed_multiplier
    
    return max(1, round(estimated_hours, 1))


st.sidebar.header("⚙️ Configuration")


mode = st.sidebar.radio("Mode", ["Single Model", "Compare Models"])

if mode == "Single Model":

    selected_model = st.sidebar.selectbox("Select Model", list(MODEL_CATALOG.keys()))
    

    selected_instance = st.sidebar.selectbox("AWS Instance", list(AWS_INSTANCES.keys()))
    

    specs = get_model_specs(selected_model)
    

    estimated_time = estimate_training_time(specs['flops'], selected_instance)
    

    st.sidebar.info(f"💡 Estimated: **{estimated_time} hrs** (100 epochs)")
    use_custom = st.sidebar.checkbox("Override estimate", value=False)
    
    if use_custom:
        training_hours = st.sidebar.slider("Custom hours", 1, 200, int(estimated_time))
    else:
        training_hours = estimated_time
    

    st.header(f"📊 {selected_model} Specifications")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Parameters", specs['params_str'])
    with col2:
        st.metric("FLOPs", specs['flops_str'])
    with col3:
        st.metric("Model Size", f"{specs['size_mb']:.2f} MB")
    with col4:
        st.metric("Architecture Depth", specs['depth'])
    

    st.subheader("🏗️ Architecture Details")
    arch_col1, arch_col2 = st.columns(2)
    
    with arch_col1:
        st.info(f"**Family:** {specs['family']}")
        st.info(f"**Type:** Convolutional Neural Network")
    with arch_col2:
        st.info(f"**Layers:** {specs['depth']}")
        st.info(f"**Input Size:** 224×224×3")
    

    st.subheader("💰 AWS SageMaker Cost Estimation")
    
    instance_info = AWS_INSTANCES[selected_instance]
    total_cost = instance_info['price'] * training_hours
    
    cost_col1, cost_col2, cost_col3 = st.columns(3)
    
    with cost_col1:
        st.metric("Instance Type", selected_instance.split('(')[0].strip())
        st.caption(f"GPU: {instance_info['gpu']}")
    with cost_col2:
        st.metric("Hourly Rate", f"${instance_info['price']:.3f}/hr")
        st.caption(f"Memory: {instance_info['memory']}")
    with cost_col3:
        st.metric("Total Cost", f"${total_cost:.2f}")
        st.caption(f"For {training_hours} hours")
    

    st.subheader("📈 Cost Breakdown")
    
    hours_range = list(range(1, 101))
    costs = [instance_info['price'] * h for h in hours_range]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours_range,
        y=costs,
        mode='lines',
        name='Cost',
        line=dict(color='#3498db', width=3),
        fill='tozeroy'
    ))
    
    fig.add_vline(x=training_hours, line_dash="dash", line_color="red", 
                  annotation_text=f"Your estimate: ${total_cost:.2f}")
    
    fig.update_layout(
        title="Training Cost vs Duration",
        xaxis_title="Training Duration (hours)",
        yaxis_title="Cost (USD)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.sidebar.subheader("Select Models to Compare")
    model1 = st.sidebar.selectbox("Model 1", list(MODEL_CATALOG.keys()), index=0)
    model2 = st.sidebar.selectbox("Model 2", list(MODEL_CATALOG.keys()), index=7)
    

    specs1 = get_model_specs(model1)
    specs2 = get_model_specs(model2)
    

    st.header(f"⚖️ Comparing {model1} vs {model2}")
    

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📊 {model1}")
        st.metric("Parameters", specs1['params_str'])
        st.metric("FLOPs", specs1['flops_str'])
        st.metric("Size", f"{specs1['size_mb']:.2f} MB")
        st.metric("Depth", specs1['depth'])
        st.info(f"**Family:** {specs1['family']}")
    
    with col2:
        st.subheader(f"📊 {model2}")
        st.metric("Parameters", specs2['params_str'])
        st.metric("FLOPs", specs2['flops_str'])
        st.metric("Size", f"{specs2['size_mb']:.2f} MB")
        st.metric("Depth", specs2['depth'])
        st.info(f"**Family:** {specs2['family']}")
    

    st.subheader("📊 Visual Comparison")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:

        fig_params = go.Figure(data=[
            go.Bar(name='Parameters', x=[model1, model2], 
                   y=[specs1['total_params']/1e6, specs2['total_params']/1e6],
                   marker_color=['#e74c3c', '#3498db'])
        ])
        fig_params.update_layout(title="Parameters (Millions)", yaxis_title="Millions")
        st.plotly_chart(fig_params, use_container_width=True)
    
    with viz_col2:

        fig_flops = go.Figure(data=[
            go.Bar(name='FLOPs', x=[model1, model2], 
                   y=[specs1['flops']/1e9, specs2['flops']/1e9],
                   marker_color=['#e74c3c', '#3498db'])
        ])
        fig_flops.update_layout(title="FLOPs (Billions)", yaxis_title="Billions")
        st.plotly_chart(fig_flops, use_container_width=True)
    

    st.subheader("🎯 Efficiency Analysis")
    
    param_ratio = specs1['total_params'] / specs2['total_params']
    flops_ratio = specs1['flops'] / specs2['flops']
    size_ratio = specs1['size_mb'] / specs2['size_mb']
    
    ratio_col1, ratio_col2, ratio_col3 = st.columns(3)
    
    with ratio_col1:
        st.metric("Parameter Ratio", f"{param_ratio:.2f}x")
        st.caption(f"{model1} / {model2}")
    with ratio_col2:
        st.metric("FLOPs Ratio", f"{flops_ratio:.2f}x")
        st.caption(f"{model1} / {model2}")
    with ratio_col3:
        st.metric("Size Ratio", f"{size_ratio:.2f}x")
        st.caption(f"{model1} / {model2}")


st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Model specs calculated using PyTorch & THOP")
