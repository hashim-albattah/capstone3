�	�wg�,c@�wg�,c@!�wg�,c@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�wg�,c@���2��?1��a�2`@A����W��?I[��X�6@*	���(\A@2F
Iterator::Model�
Y��?!      Y@)5D�o�?1H��6MO@:Preprocessing2P
Iterator::Model::Prefetchעh[͊?!�<dɲB@)עh[͊?1�<dɲB@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�14.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���2��?���2��?!���2��?      ��!       "	��a�2`@��a�2`@!��a�2`@*      ��!       2	����W��?����W��?!����W��?:	[��X�6@[��X�6@![��X�6@B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb �">
"functional_1/block_1_expand/Conv2DConv2D@���T��?!@���T��?"6
functional_1/Conv_1/Conv2DConv2D���Rth�?!�Ȓ\G-�?"@
$functional_1/block_16_project/Conv2DConv2DO�t+��?!�\�9���?"5
functional_1/Conv1/Conv2DConv2D��_�=�?!�X*�a��?"E
)functional_1/expanded_conv_project/Conv2DConv2D؉RMcղ?!�}?:�E�?"?
#functional_1/block_2_project/Conv2DConv2D�����?!K����I�?"?
#functional_1/block_1_project/Conv2DConv2DRݴ����?! ����?"@
$functional_1/block_13_project/Conv2DConv2D0uw���?!s�iTC�?"@
$functional_1/block_10_project/Conv2DConv2D��P�ё�?![��q|�?"?
#functional_1/block_3_project/Conv2DConv2Dx.~IS$�?!Bw���~�?Q      Y@YTRb�@an��sX@q�F+�[aR@yC��\�Z?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�14.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�73.5212% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 