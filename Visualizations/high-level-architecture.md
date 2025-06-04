# High-Level System Architecture

## LLM-Enhanced-DFine-Object-Detection-Framework

```mermaid
graph TB
    A[User Interface<br/>Web Frontend] --> B[Flask Application<br/>app.py]
    B --> C[Request Handler<br/>Upload & Process]
    
    C --> D[LLM Service<br/>Groq Llama AI]
    C --> E[File Manager<br/>Upload/Output Handling]
    
    D --> F[Natural Language<br/>Processing]
    F --> G[Parameter Extraction<br/>Objects, ROI, Confidence]
    
    G --> H[D-Fine Service<br/>Object Detection Engine]
    H --> I[Computer Vision<br/>Processing]
    
    I --> J{Media Type?}
    J -->|Image| K[Single Frame<br/>Detection]
    J -->|Video| L[Multi-Frame<br/>Processing]
    
    K --> M[Object Detection<br/>& Annotation]
    L --> N[Tracking Service<br/>Unique Object Tracking]
    N --> M
    
    M --> O[Result Processing<br/>& Visualization]
    O --> P[Output Generation<br/>Annotated Media]
    P --> Q[Response Handler<br/>JSON + File URLs]
    Q --> A
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style D fill:#fff3e0
    style H fill:#e8f5e8
    style I fill:#fff8e1
    style O fill:#fce4ec
```

## System Components

### 1. **Frontend Layer**
- **Web Interface**: User-friendly upload and prompt interface
- **Real-time Results**: Dynamic display of detection results

### 2. **Application Layer**
- **Flask Framework**: RESTful API and web serving
- **Request Routing**: Handle upload, processing, and result endpoints
- **Session Management**: Secure user sessions and file handling

### 3. **AI Processing Layer**
- **Groq LLM Service**: Natural language understanding
- **Parameter Extraction**: Convert text to detection parameters
- **Smart Prompt Analysis**: Object identification and ROI detection

### 4. **Computer Vision Layer**
- **D-Fine Detection**: Advanced object detection model
- **Multi-format Support**: Images and video processing
- **Tracking Integration**: Unique object tracking across frames

### 5. **Data Management Layer**
- **File Handling**: Secure upload and output management
- **Result Storage**: Temporary storage of processed media
- **Cleanup Services**: Automatic file cleanup after processing

## Key Features

- **Natural Language Interface**: Plain English detection requests
- **Multi-Modal Processing**: Images and videos
- **Real-time Processing**: Fast detection and response
- **Scalable Architecture**: Modular component design
- **Security**: Environment-based configuration 