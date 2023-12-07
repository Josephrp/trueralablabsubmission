# Multi-Modal Multi-Model Evaluation to Optimize Downstream Application Performance

## Our Solution

##### [Try The Demo For Multimed Here](https://github.com/Josephrp/trueralablabsubmission/blob/main/trythedemo.md)

## Abstract

This paper presents an approach for multimodal application performance optimization using the TruEra Machine Learning Ops platform for model evaluation. 5 Vision Models, 5 Audio Models, and 5 Text Models were evaluated based on prompting, performance, and various sequential configurations to produce the best downstream outcomes based on human evaluation. The selected configuration and prompts are available in a demo here.

## Problem Statement

Enterprise application prototypers face the problem that each model is a little bit different. Some novel functions or capabilities can also drastically improve an agent’s performance. Multimodality compounds this issue and makes assessments all the more time-consuming and technically challenging. Intel, Google Vertex, Milvus, and TruEra, provide models, model serving, retrieval-augmented generation, and evaluation, respectively. But what is their optimal configuration for a given demo?

## Literature Review

We have selected as many models as time allowed which was 7 days. Some models had endpoints in huggingface, others in Google Vertex. We also fine-tuned models described below.

### Vision Models

The following vision models were included in the study:

- **GPT4V**: A vision model based on the GPT-4 architecture, designed for image generation and analysis.
- **LLava-1.5**: An advanced image processing model known for its robust feature extraction capabilities.
- **Qwen-VL**: A vision-language model focused on understanding and generating multimodal content.
- **Clip (Google/Vertex)**: A model developed by Google, designed for image recognition and classification, leveraging Vertex AI's capabilities.
- **Fuyu-8B**: Great vision model previously retained over GPT4V for its ability to process images and available endpoint.

### Audio Models

The audio models evaluated were:

- **Seamless 1.0 & 2.0**: Two versions of an audio processing model, known for their speech recognition and audio analysis capabilities.
- **Qwen Audio**: A model specializing in audio processing and understanding.
- **Whisper2 & Whisper3**: Advanced versions of an audio model designed for speech-to-text conversion and audio analysis.
- **Seamless on device**: A variant of the Seamless model optimized for on-device applications.
- **GoogleAUDIOMODEL**: A comprehensive audio model developed by Google, known for its accuracy in speech recognition and audio processing.

### Text Models

The text models included in the study were:

- **StableMed (StableLM Finetune)**: A specialized version of StableLM, fine-tuned for medical text analysis.
- **MistralMed (Mistral Finetune)**: A fine-tuned version of the Mistral model, tailored for medical text processing.
- **Qwen On Device**: A text processing model optimized for on-device applications.
- **GPT**: The base GPT model, known for its general text generation and processing capabilities.
- **Mistral Endpoint**: A variant of the Mistral model, designed for endpoint applications.
- **Intel Neural Chat**: A text-based model developed by Intel, focusing on conversational AI.
- **BERT (Google/Vertex)**: A model developed by Google, using Vertex AI, known for its effectiveness in understanding and processing natural language.

## Methods

This study aimed to evaluate the performance of various vision, audio, and text models in producing downstream outcomes, with a focus on human-centered applications. The evaluation was structured around three main axes: prompting strategies, performance metrics, and sequential configurations of the models. The models were assessed based on their ability to process and generate information in their respective domains (vision, audio, text) and their effectiveness in integrated applications.

### Global Evaluation Criteria

The models were evaluated based on the following criteria:

- **Prompting**: The effectiveness of different prompting strategies in eliciting accurate and relevant responses from the models.
- **Performance**: Quantitative metrics such as accuracy, speed, and error rate were used to assess the performance of each model.
- **Sequential Configurations**: The models were combined in various sequential configurations to determine the most effective combinations for specific tasks.

### Human Evaluation

The ultimate measure of each model's effectiveness was based on human evaluation. A panel of n=1 with domain-specific knowledge assessed the outputs for relevance, accuracy, and utility.

### Using TruEra For Multimodal AI Application Lifecycle Evaluation

#### Initial Phase

In the early stages of application development, TruEra assists in data preparation, input selection and engineering, model architecture choice, and hyperparameter tuning. This foundational support is crucial for building robust models.

#### Model Evaluation and Improvement

Post-development, models are introduced to the TruEra platform for quality assessment. The platform's diagnostic capabilities enable users to identify and rectify weaknesses, thereby enhancing model strength and reliability.

#### Deployment and Continuous Improvement

Once a model meets the desired criteria, it is deployed into the production environment. TruEra's monitoring tools play a critical role in this phase, offering ongoing support and insights for continuous model improvement.

## Evaluation Results



| Category | Model            | Subcategory          | Evaluation Results                                                                                   |
|----------|------------------|----------------------|------------------------------------------------------------------------------------------------------|
| Vision   | GPT4V            | Image Generation     | Requires jailbreak, not effective for medical cases, prone to unavailability.                        |
|          | LLava-1.5        | Image Processing     | High quality but outperformed by Fuyu in quality metric.                                              |
|          | Qwen-VL          | Vision-Language      | Superior in quality response, versatile in features (e.g., image in response).                       |
|          | Clip (Google/Vertex) | Image Recognition  | Limited capabilities; fine-tuned models improve performance in pipelines.                            |
|          | Fuyu-8B          | Image Processing     | Previously SOTA, now surpassed by Qwen-VL.                                                           |
| Audio    | Seamless 1.0     | Audio Processing     | Previous SOTA for capability and cost.                                                               |
|          | Seamless 2.0     | Audio Processing     | Current SOTA for capability and cost (TruEra).                                                       |
|          | Qwen Audio       | Audio Processing     | Inconsistent or hallucinatory responses (TruEra).                                                    |
|          | Whisper2 & Whisper3 | Speech-to-Text     | On-device models lacking text-to-speech capability.                                                  |
|          | Seamless on device | Audio Processing   | Smaller version of Seamless without text-to-speech.                                                  |
|          | GoogleAUDIOMODEL | Speech Recognition   | Served using Vertex, lacks text-to-speech capability.                                                |
| Text     | StableMed (StableLM Finetune) | Medical Text Analysis | Previous SOTA, retained for zero marginal cost.                              |
|          | MistralMed (Mistral Finetune) | Medical Text Processing | Better performance, costly for self-hosting.                                |
|          | Qwen On Device   | Text Processing      | Retained for zero marginal cost, better performance than StableMed.                                  |
|          | GPT              | Text Generation      | Poor performance.                                                                                    |
|          | Mistral Endpoint | Text Processing      | Zero marginal cost, but MistralMed performs better.                                                  |
|          | Intel Neural Chat | Conversational AI   | Did not render quality results, costs are prohibitive.                                               |
|          | BERT (Google/Vertex) | Natural Language Understanding | Cost of serving the model is prohibitive.                               |


### Vision Models

- **GPT4V**: Requires a jailbreak to work, prone to require additional jailbreaks as censorship evolves. Did not render for most medical-related cases. Prone to unavailability.
- **LLava-1.5**: Beaten on the quality metric by Fuyu.
- **Qwen-VL**: Beat Fuyu on the quality response metric and has more capabilities that can be built into a feature: for example, an image in the response.
- **Clip (Google/Vertex)**: Clip lacked capabilities but fine-tuned CLIP models help the overall response as part of a pipeline.
- **Fuyu-8B**: Previous SOTA model. (displaced by Qwen)

### Audio Models

- **Seamless 1.0**: Previous SOTA for capability and cost.
- **Seamless 2.0**: Current SOTA for capability and cost (TruEra)
- **Qwen Audio**: Produced inconsistent or hallucinatory responses (TruEra)
- **Whisper2 & Whisper3**: Whisper are on-device models that do not have the required text-to-speech capability.
- **Seamless on device**: Smaller seamless without text-to-speech.
- **GoogleAUDIOMODEL**: Served using vertex, does not have text-to-speech.

### Text Models

- **StableMed (StableLM Finetune)**: Previous SOTA - Retained for its marginal cost of 0.
- **MistralMed (Mistral Finetune)**: Better performance, costly for self-hosting.
- **Qwen On Device**: Retained for its marginal cost of zero and better performance than StableMed.
- **GPT**: Poor performance.
- **Mistral Endpoint**: Marginal cost of zero but Mistral Med is better.
- **Intel Neural Chat**: Did not render quality results, costs are prohibitive.
- **BERT (Google/Vertex)**: Cost of serving the model is prohibitive.

## Further Research
