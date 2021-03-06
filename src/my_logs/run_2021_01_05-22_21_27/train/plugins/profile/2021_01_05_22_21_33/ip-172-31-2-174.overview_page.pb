�	�6�x�c@�6�x�c@!�6�x�c@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�6�x�c@\kF�?1����_@A� Ϡ��?I��^c7@*	+��^A@2F
Iterator::Model�!���ɡ?!      Y@)l��F���?1�M@:Preprocessing2P
Iterator::Model::Prefetch\Z�{,�?!P��=p�D@)\Z�{,�?1P��=p�D@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�15.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	\kF�?\kF�?!\kF�?      ��!       "	����_@����_@!����_@*      ��!       2	� Ϡ��?� Ϡ��?!� Ϡ��?:	��^c7@��^c7@!��^c7@B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb �">
"functional_1/block_1_expand/Conv2DConv2Dy7�q5��?!y7�q5��?"6
functional_1/Conv_1/Conv2DConv2D����f�?!^����)�?"@
$functional_1/block_16_project/Conv2DConv2D#���H��?!�U���?"5
functional_1/Conv1/Conv2DConv2D��T߮�?!	ʆ��?"E
)functional_1/expanded_conv_project/Conv2DConv2D�֭3ކ�?!���)s�?"?
#functional_1/block_2_project/Conv2DConv2Dލ�~�?!��ђ
k�?"?
#functional_1/block_1_project/Conv2DConv2D{��?!��J��?"@
$functional_1/block_13_project/Conv2DConv2D�]�F�?!cw��W�?"@
$functional_1/block_10_project/Conv2DConv2D�N��(E�?!L|��Q��?"-
IteratorGetNext/_2_Recv���7�؟?!T�] ��?Q      Y@YTRb�@an��sX@q�q��c*R@yht"�\Z?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�15.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�72.6623% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 