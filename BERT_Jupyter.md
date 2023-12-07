<img src="https://upload.wikimedia.org/wikipedia/en/6/6d/Nvidia_image_logo.svg" style="width: 90px; float: right;">

# BERT Question Answering in TensorFlow with Mixed Precision


| **Model** | **Hidden layers** | **Hidden unit size** | **Attention heads** | **Feedforward filter size** | **Max sequence length** | **Parameters** |
|:---------:|:----------:|:----:|:---:|:--------:|:---:|:----:|
|BERTLARGE|24 encoder|1024| 16|4 x 1024|512|330M|


```python
use_mixed_precision_model = True
```


```python
# bert_tf_ckpt_large_qa_squad2_amp_384
DATA_DIR_FT = '/workspace/bert/data/finetuned_large_model_SQUAD2.0'
!mkdir -p $DATA_DIR_FT
    
!wget --content-disposition -O $DATA_DIR_FT/bert_tf_ckpt_large_qa_squad2_amp_384_19.03.1.zip  \
https://api.ngc.nvidia.com/v2/models/nvidia/bert_tf_ckpt_large_qa_squad2_amp_384/versions/19.03.1/zip \
&& unzip -n -d $DATA_DIR_FT/ $DATA_DIR_FT/bert_tf_ckpt_large_qa_squad2_amp_384_19.03.1.zip \
&& rm -rf $DATA_DIR_FT/bert_tf_ckpt_large_qa_squad2_amp_384_19.03.1.zip
```

    --2023-12-06 22:43:27--  https://api.ngc.nvidia.com/v2/models/nvidia/bert_tf_ckpt_large_qa_squad2_amp_384/versions/19.03.1/zip
    Resolving api.ngc.nvidia.com (api.ngc.nvidia.com)... 34.218.164.99, 54.191.184.208
    Connecting to api.ngc.nvidia.com (api.ngc.nvidia.com)|34.218.164.99|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://prod-model-registry-ngc-bucket.s3.us-west-2.amazonaws.com/org/nvidia/models/bert_tf_ckpt_large_qa_squad2_amp_384/versions/19.03.1/files.zip?response-content-disposition=attachment%3B%20filename%3D%22files.zip%22&response-content-type=application%2Fzip&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231206T224327Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=AKIA3PSNVSIZ7SU24VXK%2F20231206%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Signature=6d35e3d5be6b233dc983035d7a1b1e94a8d842f96c770c4f8845b6d5d8b6029b [following]
    --2023-12-06 22:43:27--  https://prod-model-registry-ngc-bucket.s3.us-west-2.amazonaws.com/org/nvidia/models/bert_tf_ckpt_large_qa_squad2_amp_384/versions/19.03.1/files.zip?response-content-disposition=attachment%3B%20filename%3D%22files.zip%22&response-content-type=application%2Fzip&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231206T224327Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=AKIA3PSNVSIZ7SU24VXK%2F20231206%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Signature=6d35e3d5be6b233dc983035d7a1b1e94a8d842f96c770c4f8845b6d5d8b6029b
    Resolving prod-model-registry-ngc-bucket.s3.us-west-2.amazonaws.com (prod-model-registry-ngc-bucket.s3.us-west-2.amazonaws.com)... 3.5.86.211, 3.5.79.227, 52.92.145.74, ...
    Connecting to prod-model-registry-ngc-bucket.s3.us-west-2.amazonaws.com (prod-model-registry-ngc-bucket.s3.us-west-2.amazonaws.com)|3.5.86.211|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 3672271344 (3.4G) [application/zip]
    Saving to: ‘/workspace/bert/data/finetuned_large_model_SQUAD2.0/bert_tf_ckpt_large_qa_squad2_amp_384_19.03.1.zip’
    
    /workspace/bert/dat 100%[===================>]   3.42G  23.1MB/s    in 3m 11s  
    
    2023-12-06 22:46:39 (18.3 MB/s) - ‘/workspace/bert/data/finetuned_large_model_SQUAD2.0/bert_tf_ckpt_large_qa_squad2_amp_384_19.03.1.zip’ saved [3672271344/3672271344]
    
    Archive:  /workspace/bert/data/finetuned_large_model_SQUAD2.0/bert_tf_ckpt_large_qa_squad2_amp_384_19.03.1.zip
      inflating: /workspace/bert/data/finetuned_large_model_SQUAD2.0/bert_config.json  
      inflating: /workspace/bert/data/finetuned_large_model_SQUAD2.0/model.ckpt.data-00000-of-00001  
      inflating: /workspace/bert/data/finetuned_large_model_SQUAD2.0/model.ckpt.index  
      inflating: /workspace/bert/data/finetuned_large_model_SQUAD2.0/model.ckpt.meta  
      inflating: /workspace/bert/data/finetuned_large_model_SQUAD2.0/tf_bert_squad_1n_fp16_gbs32.190523090758.log  
      inflating: /workspace/bert/data/finetuned_large_model_SQUAD2.0/vocab.txt  




```python
# Download BERT helper scripts
!wget -nc --show-progress -O bert_scripts.zip \
     https://api.ngc.nvidia.com/v2/recipes/nvidia/bert_for_tensorflow/versions/1/zip
!mkdir -p /workspace/bert
!unzip -n -d /workspace/bert bert_scripts.zip
```

    File ‘bert_scripts.zip’ already there; not retrieving.
    Archive:  bert_scripts.zip
      inflating: /workspace/bert/.dockerignore  
      inflating: /workspace/bert/CONTRIBUTING.md  
      inflating: /workspace/bert/Dockerfile  
      inflating: /workspace/bert/LICENSE  
      inflating: /workspace/bert/NOTICE  
      inflating: /workspace/bert/README.md  
      inflating: /workspace/bert/__init__.py  
      inflating: /workspace/bert/create_pretraining_data.py  
      inflating: /workspace/bert/data/README.md  
      inflating: /workspace/bert/data/bookcorpus/clean_and_merge_text.py  
      inflating: /workspace/bert/data/bookcorpus/config.sh  
      inflating: /workspace/bert/data/bookcorpus/create_pseudo_test_set.py  
      inflating: /workspace/bert/data/bookcorpus/create_pseudo_test_set.sh  
      inflating: /workspace/bert/data/bookcorpus/preprocessing.sh  
      inflating: /workspace/bert/data/bookcorpus/preprocessing_test_set.sh  
      inflating: /workspace/bert/data/bookcorpus/preprocessing_test_set_xargs_wrapper.sh  
      inflating: /workspace/bert/data/bookcorpus/preprocessing_xargs_wrapper.sh  
      inflating: /workspace/bert/data/bookcorpus/run_preprocessing.sh  
      inflating: /workspace/bert/data/bookcorpus/sentence_segmentation_nltk.py  
      inflating: /workspace/bert/data/bookcorpus/shard_text_input_file.py  
      inflating: /workspace/bert/data/pretrained_models_google/download_models.py  
      inflating: /workspace/bert/data/squad/squad_download.sh  
      inflating: /workspace/bert/data/wikipedia_corpus/config.sh  
      inflating: /workspace/bert/data/wikipedia_corpus/create_pseudo_test_set.py  
      inflating: /workspace/bert/data/wikipedia_corpus/create_pseudo_test_set.sh  
      inflating: /workspace/bert/data/wikipedia_corpus/preprocessing.sh  
      inflating: /workspace/bert/data/wikipedia_corpus/preprocessing_test_set.sh  
      inflating: /workspace/bert/data/wikipedia_corpus/preprocessing_test_set_xargs_wrapper.sh  
      inflating: /workspace/bert/data/wikipedia_corpus/preprocessing_xargs_wrapper.sh  
      inflating: /workspace/bert/data/wikipedia_corpus/remove_tags_and_clean.py  
      inflating: /workspace/bert/data/wikipedia_corpus/run_preprocessing.sh  
      inflating: /workspace/bert/data/wikipedia_corpus/shard_text_input_file.py  
      inflating: /workspace/bert/data/wikipedia_corpus/wiki_sentence_segmentation_nltk.py  
      inflating: /workspace/bert/data/wikipedia_corpus/wiki_sentence_segmentation_spacy.py  
      inflating: /workspace/bert/data/wikipedia_corpus/wiki_sentence_segmentation_spacy_pipe.py  
      inflating: /workspace/bert/extract_features.py  
      inflating: /workspace/bert/gpu_environment.py  
      inflating: /workspace/bert/modeling.py  
      inflating: /workspace/bert/modeling_test.py  
      inflating: /workspace/bert/multilingual.md  
      inflating: /workspace/bert/optimization.py  
      inflating: /workspace/bert/optimization_test.py  
      inflating: /workspace/bert/predicting_movie_reviews_with_bert_on_tf_hub.ipynb  
      inflating: /workspace/bert/requirements.txt  
      inflating: /workspace/bert/run_classifier.py  
      inflating: /workspace/bert/run_classifier_with_tfhub.py  
      inflating: /workspace/bert/run_pretraining.py  
      inflating: /workspace/bert/run_squad.py  
      inflating: /workspace/bert/sample_text.txt  
      inflating: /workspace/bert/scripts/data_download.sh  
      inflating: /workspace/bert/scripts/data_download_helper.sh  
      inflating: /workspace/bert/scripts/docker/build.sh  
      inflating: /workspace/bert/scripts/docker/launch.sh  
      inflating: /workspace/bert/scripts/finetune_inference_benchmark.sh  
      inflating: /workspace/bert/scripts/finetune_train_benchmark.sh  
      inflating: /workspace/bert/scripts/run.sub  
      inflating: /workspace/bert/scripts/run_pretraining.sh  
      inflating: /workspace/bert/scripts/run_squad.sh  
      inflating: /workspace/bert/scripts/run_squad_inference.sh  
      inflating: /workspace/bert/scripts/start_pretraining.sh  
      inflating: /workspace/bert/tokenization.py  
      inflating: /workspace/bert/tokenization_test.py  


### BERT Config


```python
# Download BERT vocab file
!mkdir -p /workspace/bert/config.qa
!wget -nc https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt \
    -O /workspace/bert/config.qa/vocab.txt
```

    --2023-12-06 22:47:40--  https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
    Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.217.192.128, 16.182.32.168, 52.216.112.77, ...
    Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.217.192.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 231508 (226K) [text/plain]
    Saving to: ‘/workspace/bert/config.qa/vocab.txt’
    
    /workspace/bert/con 100%[===================>] 226.08K  1.18MB/s    in 0.2s    
    
    2023-12-06 22:47:41 (1.18 MB/s) - ‘/workspace/bert/config.qa/vocab.txt’ saved [231508/231508]
    



```python
%%writefile /workspace/bert/config.qa/bert_config.json
{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "max_position_embeddings": 512,
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```

    Writing /workspace/bert/config.qa/bert_config.json


### Helper Functions


```python
# Create dynamic JSON files based on user inputs
def write_input_file(context, qinputs, predict_file):
    # Remove quotes and new lines from text for valid JSON
    context = context.replace('"', '').replace('\n', '')
    # Create JSON dict to write
    json_dict = {
      "data": [
        {
          "title": "BERT QA",
          "paragraphs": [
            {
              "context": context,
              "qas": qinputs
            }
          ]
        }
      ]
    }
    # Write JSON to input file
    with open(predict_file, 'w') as json_file:
        import json
        json.dump(json_dict, json_file, indent=2)
    
# Display Inference Results as HTML Table
def display_results(predict_file, output_prediction_file):
    import json
    from IPython.display import display, HTML

    # Here we show only the prediction results, nbest prediction is also available in the output directory
    results = ""
    with open(predict_file, 'r') as query_file:
        queries = json.load(query_file)
        input_data = queries["data"]
        with open(output_prediction_file, 'r') as result_file:
            data = json.load(result_file)
            for entry in input_data:
                for paragraph in entry["paragraphs"]:
                    for qa in paragraph["qas"]:
                        results += "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(qa["id"], qa["question"], data[qa["id"]])

    display(HTML("<table><tr><th>Id</th><th>Question</th><th>Answer</th></tr>{}</table>".format(results)))
```

## 3. BERT Inference: Question Answering



```python
# Create BERT input file with (1) context and (2) questions to be answered based on that context
predict_file = '/workspace/bert/config.qa/input.json'
```


```python
%%writefile $predict_file
{"data": 
 [
     {"title": "Project Apollo",
      "paragraphs": [
          {"context":"The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which accomplished landing the first humans on the Moon from 1969 to 1972. First conceived during Dwight D. Eisenhower's administration as a three-man spacecraft to follow the one-man Project Mercury which put the first Americans in space, Apollo was later dedicated to President John F. Kennedy's national goal of landing a man on the Moon and returning him safely to the Earth by the end of the 1960s, which he proposed in a May 25, 1961, address to Congress. Project Mercury was followed by the two-man Project Gemini. The first manned flight of Apollo was in 1968. Apollo ran from 1961 to 1972, and was supported by the two man Gemini program which ran concurrently with it from 1962 to 1966. Gemini missions developed some of the space travel techniques that were necessary for the success of the Apollo missions. Apollo used Saturn family rockets as launch vehicles. Apollo/Saturn vehicles were also used for an Apollo Applications Program, which consisted of Skylab, a space station that supported three manned missions in 1973-74, and the Apollo-Soyuz Test Project, a joint Earth orbit mission with the Soviet Union in 1975.", 
           "qas": [
               { "question": "What project put the first Americans into space?", 
                 "id": "Q1"
               },
               { "question": "What program was created to carry out these projects and missions?",
                 "id": "Q2"
               },
               { "question": "What year did the first manned Apollo flight occur?",
                 "id": "Q3"
               },                
               { "question": "What President is credited with the notion of putting Americans on the moon?",
                 "id": "Q4"
               },
               { "question": "Who did the U.S. collaborate with on an Earth orbit mission in 1975?",
                 "id": "Q5"
               },
               { "question": "How long did Project Apollo run?",
                 "id": "Q6"
               },               
               { "question": "What program helped develop space travel techniques that Project Apollo used?",
                 "id": "Q7"
               },                
               {"question": "What space station supported three manned missions in 1973-1974?",
                 "id": "Q8"
               }
]}]}]}
```

    Writing /workspace/bert/config.qa/input.json


```python
import os

# This specifies the model architecture.
bert_config_file = '/workspace/bert/config.qa/bert_config.json'

# The vocabulary file that the BERT model was trained on.
vocab_file = '/workspace/bert/config.qa/vocab.txt'

# Initiate checkpoint to the fine-tuned BERT Large model
init_checkpoint = os.path.join('/workspace/bert/data/finetuned_large_model_SQUAD2.0/model.ckpt')

# Create the output directory where all the results are saved.
output_dir = '/workspace/bert/results'
output_prediction_file = os.path.join(output_dir,'predictions.json')
    
# Whether to lower case the input - True for uncased models / False for cased models.
do_lower_case = True
  
# Total batch size for predictions
predict_batch_size = 8

# Whether to run eval on the dev set.
do_predict = True

# When splitting up a long document into chunks, how much stride to take between chunks.
doc_stride = 128

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
max_seq_length = 384
```

### 4a. Run Inference


```python
# Ask BERT questions
!python /workspace/bert/run_squad.py \
  --bert_config_file=$bert_config_file \
  --vocab_file=$vocab_file \
  --init_checkpoint=$init_checkpoint \
  --output_dir=$output_dir \
  --do_predict=$do_predict \
  --predict_file=$predict_file \
  --predict_batch_size=$predict_batch_size \
  --doc_stride=$doc_stride \
  --max_seq_length=$max_seq_length
```

    2023-12-06 22:47:47.594956: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
    WARNING:tensorflow:Deprecation warnings have been disabled. Set TF_ENABLE_DEPRECATION_WARNINGS=1 to re-enable them.
    WARNING:tensorflow:From /workspace/bert/optimization.py:110: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:162: The name tf.train.SessionRunHook is deprecated. Please use tf.estimator.SessionRunHook instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:1409: The name tf.app.run is deprecated. Please use tf.compat.v1.app.run instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:1174: The name tf.logging.set_verbosity is deprecated. Please use tf.compat.v1.logging.set_verbosity instead.
    
    W1206 22:47:55.698684 140392291764032 module_wrapper.py:137] From /workspace/bert/run_squad.py:1174: The name tf.logging.set_verbosity is deprecated. Please use tf.compat.v1.logging.set_verbosity instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:1174: The name tf.logging.INFO is deprecated. Please use tf.compat.v1.logging.INFO instead.
    
    W1206 22:47:55.699086 140392291764032 module_wrapper.py:137] From /workspace/bert/run_squad.py:1174: The name tf.logging.INFO is deprecated. Please use tf.compat.v1.logging.INFO instead.
    
    WARNING:tensorflow:From /workspace/bert/modeling.py:94: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    W1206 22:47:55.699397 140392291764032 module_wrapper.py:137] From /workspace/bert/modeling.py:94: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:1183: The name tf.gfile.MakeDirs is deprecated. Please use tf.io.gfile.makedirs instead.
    
    W1206 22:47:55.700649 140392291764032 module_wrapper.py:137] From /workspace/bert/run_squad.py:1183: The name tf.gfile.MakeDirs is deprecated. Please use tf.io.gfile.makedirs instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:1199: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
    
    W1206 22:47:55.845726 140392291764032 module_wrapper.py:137] From /workspace/bert/run_squad.py:1199: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
    
    WARNING:tensorflow:
    The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
      * https://github.com/tensorflow/io (for I/O related ops)
    If you depend on functionality not listed there, please file an issue.
    
    W1206 22:47:55.846174 140392291764032 lazy_loader.py:50] 
    The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
      * https://github.com/tensorflow/io (for I/O related ops)
    If you depend on functionality not listed there, please file an issue.
    
    WARNING:tensorflow:TensorFlow will not use sklearn by default. This improves performance in some cases. To enable sklearn export the environment variable  TF_ALLOW_IOLIBS=1.
    W1206 22:47:56.334908 140392291764032 metric_loss_ops.py:43] TensorFlow will not use sklearn by default. This improves performance in some cases. To enable sklearn export the environment variable  TF_ALLOW_IOLIBS=1.
    WARNING:tensorflow:TensorFlow will not use Dask by default. This improves performance in some cases. To enable Dask export the environment variable  TF_ALLOW_IOLIBS=1.
    W1206 22:47:56.599967 140392291764032 dask_io.py:42] TensorFlow will not use Dask by default. This improves performance in some cases. To enable Dask export the environment variable  TF_ALLOW_IOLIBS=1.
    WARNING:tensorflow:TensorFlow will not use Pandas by default. This improves performance in some cases. To enable Pandas export the environment variable  TF_ALLOW_IOLIBS=1.
    W1206 22:47:56.637463 140392291764032 pandas_io.py:43] TensorFlow will not use Pandas by default. This improves performance in some cases. To enable Pandas export the environment variable  TF_ALLOW_IOLIBS=1.
    WARNING:tensorflow:Estimator's model_fn (<function model_fn_builder.<locals>.model_fn at 0x7faf1f089ee0>) includes params argument, but params are not passed to Estimator.
    W1206 22:47:58.596829 140392291764032 estimator.py:1992] Estimator's model_fn (<function model_fn_builder.<locals>.model_fn at 0x7faf1f089ee0>) includes params argument, but params are not passed to Estimator.
    INFO:tensorflow:Using config: {'_model_dir': '/workspace/bert/results', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': 1000, '_save_checkpoints_secs': None, '_session_config': , '_keep_checkpoint_max': 1, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': None, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7faf17b32a30>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1, '_tpu_config': TPUConfig(iterations_per_loop=1000, num_shards=8, num_cores_per_replica=None, per_host_input_for_training=3, tpu_job_name=None, initial_infeed_sleep_secs=None, input_partition_dims=None, eval_training_input_configuration=2, experimental_host_call_every_n_steps=1), '_cluster': None}
    I1206 22:47:58.597689 140392291764032 estimator.py:212] Using config: {'_model_dir': '/workspace/bert/results', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': 1000, '_save_checkpoints_secs': None, '_session_config': , '_keep_checkpoint_max': 1, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': None, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7faf17b32a30>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1, '_tpu_config': TPUConfig(iterations_per_loop=1000, num_shards=8, num_cores_per_replica=None, per_host_input_for_training=3, tpu_job_name=None, initial_infeed_sleep_secs=None, input_partition_dims=None, eval_training_input_configuration=2, experimental_host_call_every_n_steps=1), '_cluster': None}
    INFO:tensorflow:_TPUContext: eval_on_tpu True
    I1206 22:47:58.598125 140392291764032 tpu_context.py:220] _TPUContext: eval_on_tpu True
    WARNING:tensorflow:eval_on_tpu ignored because use_tpu is False.
    W1206 22:47:58.598366 140392291764032 tpu_context.py:222] eval_on_tpu ignored because use_tpu is False.
    WARNING:tensorflow:From /workspace/bert/run_squad.py:266: The name tf.gfile.Open is deprecated. Please use tf.io.gfile.GFile instead.
    
    W1206 22:47:58.598615 140392291764032 module_wrapper.py:137] From /workspace/bert/run_squad.py:266: The name tf.gfile.Open is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:1112: The name tf.python_io.TFRecordWriter is deprecated. Please use tf.io.TFRecordWriter instead.
    
    W1206 22:47:58.600829 140392291764032 module_wrapper.py:137] From /workspace/bert/run_squad.py:1112: The name tf.python_io.TFRecordWriter is deprecated. Please use tf.io.TFRecordWriter instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:1354: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.
    
    W1206 22:47:58.690616 140392291764032 module_wrapper.py:137] From /workspace/bert/run_squad.py:1354: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.
    
    INFO:tensorflow:***** Running predictions *****
    I1206 22:47:58.690871 140392291764032 run_squad.py:1354] ***** Running predictions *****
    INFO:tensorflow:  Num orig examples = 8
    I1206 22:47:58.690988 140392291764032 run_squad.py:1355]   Num orig examples = 8
    INFO:tensorflow:  Num split examples = 8
    I1206 22:47:58.691318 140392291764032 run_squad.py:1356]   Num split examples = 8
    INFO:tensorflow:  Batch size = 8
    I1206 22:47:58.691468 140392291764032 run_squad.py:1357]   Batch size = 8
    WARNING:tensorflow:From /workspace/bert/run_squad.py:731: The name tf.FixedLenFeature is deprecated. Please use tf.io.FixedLenFeature instead.
    
    W1206 22:47:58.691680 140392291764032 module_wrapper.py:137] From /workspace/bert/run_squad.py:731: The name tf.FixedLenFeature is deprecated. Please use tf.io.FixedLenFeature instead.
    
    INFO:tensorflow:Could not find trained model in model_dir: /workspace/bert/results, running initialization to predict.
    I1206 22:47:58.692279 140392291764032 estimator.py:614] Could not find trained model in model_dir: /workspace/bert/results, running initialization to predict.
    WARNING:tensorflow:From /usr/local/lib/python3.8/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.parse_single_example is deprecated. Please use tf.io.parse_single_example instead.
    
    W1206 22:47:59.090225 140392291764032 module_wrapper.py:137] From /usr/local/lib/python3.8/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.parse_single_example is deprecated. Please use tf.io.parse_single_example instead.
    
    INFO:tensorflow:Calling model_fn.
    I1206 22:47:59.290021 140392291764032 estimator.py:1148] Calling model_fn.
    INFO:tensorflow:Running infer on CPU
    I1206 22:47:59.290362 140392291764032 tpu_estimator.py:3124] Running infer on CPU
    WARNING:tensorflow:From /workspace/bert/modeling.py:175: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.
    
    W1206 22:47:59.323299 140392291764032 module_wrapper.py:137] From /workspace/bert/modeling.py:175: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.
    
    WARNING:tensorflow:From /workspace/bert/modeling.py:413: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.
    
    W1206 22:47:59.324956 140392291764032 module_wrapper.py:137] From /workspace/bert/modeling.py:413: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.
    
    WARNING:tensorflow:From /workspace/bert/modeling.py:494: The name tf.assert_less_equal is deprecated. Please use tf.compat.v1.assert_less_equal instead.
    
    W1206 22:47:59.352458 140392291764032 module_wrapper.py:137] From /workspace/bert/modeling.py:494: The name tf.assert_less_equal is deprecated. Please use tf.compat.v1.assert_less_equal instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:655: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.
    
    W1206 22:48:03.304627 140392291764032 module_wrapper.py:137] From /workspace/bert/run_squad.py:655: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:670: The name tf.train.init_from_checkpoint is deprecated. Please use tf.compat.v1.train.init_from_checkpoint instead.
    
    W1206 22:48:03.387527 140392291764032 module_wrapper.py:137] From /workspace/bert/run_squad.py:670: The name tf.train.init_from_checkpoint is deprecated. Please use tf.compat.v1.train.init_from_checkpoint instead.
    
    INFO:tensorflow:Done calling model_fn.
    I1206 22:48:04.450126 140392291764032 estimator.py:1150] Done calling model_fn.
    INFO:tensorflow:Graph was finalized.
    I1206 22:48:05.677110 140392291764032 monitored_session.py:240] Graph was finalized.
    2023-12-06 22:48:05.999260: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2199995000 Hz
    2023-12-06 22:48:06.029811: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x591ed60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2023-12-06 22:48:06.029881: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2023-12-06 22:48:06.086319: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
    2023-12-06 22:48:06.086387: E tensorflow/stream_executor/cuda/cuda_driver.cc:282] failed call to cuInit: UNKNOWN ERROR (303)
    2023-12-06 22:48:06.086420: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (3c440db9f685): /proc/driver/nvidia/version does not exist
    INFO:tensorflow:Running local_init_op.
    I1206 22:48:09.569765 140392291764032 session_manager.py:500] Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    I1206 22:48:09.756637 140392291764032 session_manager.py:502] Done running local_init_op.
    INFO:tensorflow:Processing example: 0
    I1206 22:48:36.431913 140392291764032 run_squad.py:1373] Processing example: 0
    INFO:tensorflow:prediction_loop marked as finished
    I1206 22:48:36.909057 140392291764032 error_handling.py:101] prediction_loop marked as finished
    INFO:tensorflow:prediction_loop marked as finished
    I1206 22:48:36.909364 140392291764032 error_handling.py:101] prediction_loop marked as finished
    INFO:tensorflow:-----------------------------
    I1206 22:48:36.909561 140392291764032 run_squad.py:1388] -----------------------------
    INFO:tensorflow:0 Total Inference Time = 38.22 Inference Time W/O start up overhead = 25.58 Sentences processed = 8
    I1206 22:48:36.909683 140392291764032 run_squad.py:1389] 0 Total Inference Time = 38.22 Inference Time W/O start up overhead = 25.58 Sentences processed = 8
    INFO:tensorflow:0 Inference Performance = 0.3127 sentences/sec
    I1206 22:48:36.909785 140392291764032 run_squad.py:1392] 0 Inference Performance = 0.3127 sentences/sec
    INFO:tensorflow:-----------------------------
    I1206 22:48:36.909894 140392291764032 run_squad.py:1393] -----------------------------
    INFO:tensorflow:Writing predictions to: /workspace/bert/results/predictions.json
    I1206 22:48:36.910044 140392291764032 run_squad.py:792] Writing predictions to: /workspace/bert/results/predictions.json
    INFO:tensorflow:Writing nbest to: /workspace/bert/results/nbest_predictions.json
    I1206 22:48:36.910143 140392291764032 run_squad.py:793] Writing nbest to: /workspace/bert/results/nbest_predictions.json


```python
display_results(predict_file, output_prediction_file)
```


1. Copy and paste your context from Wikipedia, news articles, etc. when prompted below
2. Enter questions based on the context when prompted below.
3. Run the inference script
4. Display the inference results


```python
predict_file = '/workspace/bert/config.qa/custom_input.json'
num_questions = 3           # You can configure this number
```


```python
# Create your own context to ask questions about.
context = input("Paste your context here: ")
```

    Paste your context here:  "Janet(s)" is the tenth episode of the third season of The Good Place. Written by Josh Siegal and Dylan Morgan and directed by Morgan Sackett, it originally aired on NBC on December 6, 2018. The episode sees Eleanor, Chidi, Tahani, and Jason accidentally transformed into versions of Janet, all played by D'Arcy Carden (pictured). Meanwhile, Michael (Ted Danson) and the real Janet (Carden) investigate if the afterlife system that sorts good and bad acts has been manipulated. Rehearsals for the episode began earlier than usual so Carden could learn to play the other characters. The episode required more visual effects than previous episodes. "Janet(s)" was watched by 2.58 million Americans in its original broadcast and was well received by critics; Carden's performance was widely praised. Themes covered include the meaning of the self, which the writers had studied in preparation. It was nominated for a Primetime Emmy Award for writing and won a Hugo Award.



```python
# Get questions from user input
questions = [input("Question {}/{}: ".format(i+1, num_questions)) for i in range(num_questions)]
# Format questions and write to JSON input file
qinputs = [{ "question":q, "id":"Q{}".format(i+1)} for i,q in enumerate(questions)]
write_input_file(context, qinputs, predict_file)
```

    Question 1/3:  what is Janets ?
    Question 2/3:  what's so special about it's rehearsals?
    Question 3/3:  what is interesting about this episode ?



```python
# Ask BERT questions
!python /workspace/bert/run_squad.py \
  --bert_config_file=$bert_config_file \
  --vocab_file=$vocab_file \
  --init_checkpoint=$init_checkpoint \
  --output_dir=$output_dir \
  --do_predict=$do_predict \
  --predict_file=$predict_file \
  --predict_batch_size=$predict_batch_size \
  --doc_stride=$doc_stride \
  --max_seq_length=$max_seq_length
```

    2023-12-06 22:49:58.431274: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
    WARNING:tensorflow:Deprecation warnings have been disabled. Set TF_ENABLE_DEPRECATION_WARNINGS=1 to re-enable them.
    WARNING:tensorflow:From /workspace/bert/optimization.py:110: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:162: The name tf.train.SessionRunHook is deprecated. Please use tf.estimator.SessionRunHook instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:1409: The name tf.app.run is deprecated. Please use tf.compat.v1.app.run instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:1174: The name tf.logging.set_verbosity is deprecated. Please use tf.compat.v1.logging.set_verbosity instead.
    
    W1206 22:49:59.753892 140708228085568 module_wrapper.py:137] From /workspace/bert/run_squad.py:1174: The name tf.logging.set_verbosity is deprecated. Please use tf.compat.v1.logging.set_verbosity instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:1174: The name tf.logging.INFO is deprecated. Please use tf.compat.v1.logging.INFO instead.
    
    W1206 22:49:59.754088 140708228085568 module_wrapper.py:137] From /workspace/bert/run_squad.py:1174: The name tf.logging.INFO is deprecated. Please use tf.compat.v1.logging.INFO instead.
    
    WARNING:tensorflow:From /workspace/bert/modeling.py:94: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    W1206 22:49:59.754268 140708228085568 module_wrapper.py:137] From /workspace/bert/modeling.py:94: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:1183: The name tf.gfile.MakeDirs is deprecated. Please use tf.io.gfile.makedirs instead.
    
    W1206 22:49:59.755285 140708228085568 module_wrapper.py:137] From /workspace/bert/run_squad.py:1183: The name tf.gfile.MakeDirs is deprecated. Please use tf.io.gfile.makedirs instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:1199: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
    
    W1206 22:49:59.900034 140708228085568 module_wrapper.py:137] From /workspace/bert/run_squad.py:1199: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
    
    WARNING:tensorflow:
    The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
      * https://github.com/tensorflow/io (for I/O related ops)
    If you depend on functionality not listed there, please file an issue.
    
    W1206 22:49:59.900435 140708228085568 lazy_loader.py:50] 
    The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
      * https://github.com/tensorflow/io (for I/O related ops)
    If you depend on functionality not listed there, please file an issue.
    
    WARNING:tensorflow:TensorFlow will not use sklearn by default. This improves performance in some cases. To enable sklearn export the environment variable  TF_ALLOW_IOLIBS=1.
    W1206 22:49:59.960247 140708228085568 metric_loss_ops.py:43] TensorFlow will not use sklearn by default. This improves performance in some cases. To enable sklearn export the environment variable  TF_ALLOW_IOLIBS=1.
    WARNING:tensorflow:TensorFlow will not use Dask by default. This improves performance in some cases. To enable Dask export the environment variable  TF_ALLOW_IOLIBS=1.
    W1206 22:50:00.017516 140708228085568 dask_io.py:42] TensorFlow will not use Dask by default. This improves performance in some cases. To enable Dask export the environment variable  TF_ALLOW_IOLIBS=1.
    WARNING:tensorflow:TensorFlow will not use Pandas by default. This improves performance in some cases. To enable Pandas export the environment variable  TF_ALLOW_IOLIBS=1.
    W1206 22:50:00.024559 140708228085568 pandas_io.py:43] TensorFlow will not use Pandas by default. This improves performance in some cases. To enable Pandas export the environment variable  TF_ALLOW_IOLIBS=1.
    WARNING:tensorflow:Estimator's model_fn (<function model_fn_builder.<locals>.model_fn at 0x7ff8ae4e7ee0>) includes params argument, but params are not passed to Estimator.
    W1206 22:50:00.400725 140708228085568 estimator.py:1992] Estimator's model_fn (<function model_fn_builder.<locals>.model_fn at 0x7ff8ae4e7ee0>) includes params argument, but params are not passed to Estimator.
    INFO:tensorflow:Using config: {'_model_dir': '/workspace/bert/results', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': 1000, '_save_checkpoints_secs': None, '_session_config': , '_keep_checkpoint_max': 1, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': None, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7ff8a6f8d9d0>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1, '_tpu_config': TPUConfig(iterations_per_loop=1000, num_shards=8, num_cores_per_replica=None, per_host_input_for_training=3, tpu_job_name=None, initial_infeed_sleep_secs=None, input_partition_dims=None, eval_training_input_configuration=2, experimental_host_call_every_n_steps=1), '_cluster': None}
    I1206 22:50:00.401468 140708228085568 estimator.py:212] Using config: {'_model_dir': '/workspace/bert/results', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': 1000, '_save_checkpoints_secs': None, '_session_config': , '_keep_checkpoint_max': 1, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': None, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7ff8a6f8d9d0>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1, '_tpu_config': TPUConfig(iterations_per_loop=1000, num_shards=8, num_cores_per_replica=None, per_host_input_for_training=3, tpu_job_name=None, initial_infeed_sleep_secs=None, input_partition_dims=None, eval_training_input_configuration=2, experimental_host_call_every_n_steps=1), '_cluster': None}
    INFO:tensorflow:_TPUContext: eval_on_tpu True
    I1206 22:50:00.401818 140708228085568 tpu_context.py:220] _TPUContext: eval_on_tpu True
    WARNING:tensorflow:eval_on_tpu ignored because use_tpu is False.
    W1206 22:50:00.401977 140708228085568 tpu_context.py:222] eval_on_tpu ignored because use_tpu is False.
    WARNING:tensorflow:From /workspace/bert/run_squad.py:266: The name tf.gfile.Open is deprecated. Please use tf.io.gfile.GFile instead.
    
    W1206 22:50:00.402160 140708228085568 module_wrapper.py:137] From /workspace/bert/run_squad.py:266: The name tf.gfile.Open is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:1112: The name tf.python_io.TFRecordWriter is deprecated. Please use tf.io.TFRecordWriter instead.
    
    W1206 22:50:00.403885 140708228085568 module_wrapper.py:137] From /workspace/bert/run_squad.py:1112: The name tf.python_io.TFRecordWriter is deprecated. Please use tf.io.TFRecordWriter instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:1354: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.
    
    W1206 22:50:00.430145 140708228085568 module_wrapper.py:137] From /workspace/bert/run_squad.py:1354: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.
    
    INFO:tensorflow:***** Running predictions *****
    I1206 22:50:00.430337 140708228085568 run_squad.py:1354] ***** Running predictions *****
    INFO:tensorflow:  Num orig examples = 3
    I1206 22:50:00.430426 140708228085568 run_squad.py:1355]   Num orig examples = 3
    INFO:tensorflow:  Num split examples = 3
    I1206 22:50:00.430613 140708228085568 run_squad.py:1356]   Num split examples = 3
    INFO:tensorflow:  Batch size = 8
    I1206 22:50:00.430695 140708228085568 run_squad.py:1357]   Batch size = 8
    WARNING:tensorflow:From /workspace/bert/run_squad.py:731: The name tf.FixedLenFeature is deprecated. Please use tf.io.FixedLenFeature instead.
    
    W1206 22:50:00.430899 140708228085568 module_wrapper.py:137] From /workspace/bert/run_squad.py:731: The name tf.FixedLenFeature is deprecated. Please use tf.io.FixedLenFeature instead.
    
    INFO:tensorflow:Could not find trained model in model_dir: /workspace/bert/results, running initialization to predict.
    I1206 22:50:00.431348 140708228085568 estimator.py:614] Could not find trained model in model_dir: /workspace/bert/results, running initialization to predict.
    WARNING:tensorflow:From /usr/local/lib/python3.8/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.parse_single_example is deprecated. Please use tf.io.parse_single_example instead.
    
    W1206 22:50:00.550237 140708228085568 module_wrapper.py:137] From /usr/local/lib/python3.8/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.parse_single_example is deprecated. Please use tf.io.parse_single_example instead.
    
    INFO:tensorflow:Calling model_fn.
    I1206 22:50:00.717918 140708228085568 estimator.py:1148] Calling model_fn.
    INFO:tensorflow:Running infer on CPU
    I1206 22:50:00.718261 140708228085568 tpu_estimator.py:3124] Running infer on CPU
    WARNING:tensorflow:From /workspace/bert/modeling.py:175: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.
    
    W1206 22:50:00.721667 140708228085568 module_wrapper.py:137] From /workspace/bert/modeling.py:175: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.
    
    WARNING:tensorflow:From /workspace/bert/modeling.py:413: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.
    
    W1206 22:50:00.723114 140708228085568 module_wrapper.py:137] From /workspace/bert/modeling.py:413: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.
    
    WARNING:tensorflow:From /workspace/bert/modeling.py:494: The name tf.assert_less_equal is deprecated. Please use tf.compat.v1.assert_less_equal instead.
    
    W1206 22:50:00.751279 140708228085568 module_wrapper.py:137] From /workspace/bert/modeling.py:494: The name tf.assert_less_equal is deprecated. Please use tf.compat.v1.assert_less_equal instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:655: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.
    
    W1206 22:50:05.186194 140708228085568 module_wrapper.py:137] From /workspace/bert/run_squad.py:655: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.
    
    WARNING:tensorflow:From /workspace/bert/run_squad.py:670: The name tf.train.init_from_checkpoint is deprecated. Please use tf.compat.v1.train.init_from_checkpoint instead.
    
    W1206 22:50:05.198187 140708228085568 module_wrapper.py:137] From /workspace/bert/run_squad.py:670: The name tf.train.init_from_checkpoint is deprecated. Please use tf.compat.v1.train.init_from_checkpoint instead.
    
    INFO:tensorflow:Done calling model_fn.
    I1206 22:50:06.235777 140708228085568 estimator.py:1150] Done calling model_fn.
    INFO:tensorflow:Graph was finalized.
    I1206 22:50:07.150721 140708228085568 monitored_session.py:240] Graph was finalized.
    2023-12-06 22:50:07.167182: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2199995000 Hz
    2023-12-06 22:50:07.168240: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x44a5b70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2023-12-06 22:50:07.168306: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2023-12-06 22:50:07.170462: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
    2023-12-06 22:50:07.170505: E tensorflow/stream_executor/cuda/cuda_driver.cc:282] failed call to cuInit: UNKNOWN ERROR (303)
    2023-12-06 22:50:07.170537: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (3c440db9f685): /proc/driver/nvidia/version does not exist
    INFO:tensorflow:Running local_init_op.
    I1206 22:50:10.139878 140708228085568 session_manager.py:500] Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    I1206 22:50:10.327550 140708228085568 session_manager.py:502] Done running local_init_op.
    INFO:tensorflow:Processing example: 0
    I1206 22:50:20.105874 140708228085568 run_squad.py:1373] Processing example: 0
    INFO:tensorflow:prediction_loop marked as finished
    I1206 22:50:20.274305 140708228085568 error_handling.py:101] prediction_loop marked as finished
    INFO:tensorflow:prediction_loop marked as finished
    I1206 22:50:20.274605 140708228085568 error_handling.py:101] prediction_loop marked as finished
    INFO:tensorflow:-----------------------------
    I1206 22:50:20.274770 140708228085568 run_squad.py:1388] -----------------------------
    INFO:tensorflow:0 Total Inference Time = 19.84 Inference Time W/O start up overhead = 8.84 Sentences processed = 8
    I1206 22:50:20.274912 140708228085568 run_squad.py:1389] 0 Total Inference Time = 19.84 Inference Time W/O start up overhead = 8.84 Sentences processed = 8
    INFO:tensorflow:0 Inference Performance = 0.9051 sentences/sec
    I1206 22:50:20.275020 140708228085568 run_squad.py:1392] 0 Inference Performance = 0.9051 sentences/sec
    INFO:tensorflow:-----------------------------
    I1206 22:50:20.275110 140708228085568 run_squad.py:1393] -----------------------------
    INFO:tensorflow:Writing predictions to: /workspace/bert/results/predictions.json
    I1206 22:50:20.275270 140708228085568 run_squad.py:792] Writing predictions to: /workspace/bert/results/predictions.json
    INFO:tensorflow:Writing nbest to: /workspace/bert/results/nbest_predictions.json
    I1206 22:50:20.275372 140708228085568 run_squad.py:793] Writing nbest to: /workspace/bert/results/nbest_predictions.json



```python
display_results(predict_file, output_prediction_file)
```


<table><tr><th>Id</th><th>Question</th><th>Answer</th></tr><tr><td>Q1</td><td>what is Janets ?</td><td>afterlife</td></tr><tr><td>Q2</td><td>what's so special about it's rehearsals?</td><td>so Carden could learn to play the other characters</td></tr><tr><td>Q3</td><td>what is interesting about this episode ?</td><td>The episode sees Eleanor, Chidi, Tahani, and Jason accidentally transformed into versions of Janet</td></tr></table>



```python

```
