чн
к¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8Я╝
|
dense_100/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*!
shared_namedense_100/kernel
u
$dense_100/kernel/Read/ReadVariableOpReadVariableOpdense_100/kernel*
_output_shapes

:d*
dtype0
t
dense_100/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_100/bias
m
"dense_100/bias/Read/ReadVariableOpReadVariableOpdense_100/bias*
_output_shapes
:d*
dtype0
|
dense_101/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*!
shared_namedense_101/kernel
u
$dense_101/kernel/Read/ReadVariableOpReadVariableOpdense_101/kernel*
_output_shapes

:d*
dtype0
t
dense_101/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_101/bias
m
"dense_101/bias/Read/ReadVariableOpReadVariableOpdense_101/bias*
_output_shapes
:*
dtype0

NoOpNoOp
╕
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*є
valueщBц B▀
╜
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

	kernel

bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api

	0

1
2
3

	0

1
2
3
 
н
layer_regularization_losses
	variables
metrics

layers
non_trainable_variables
layer_metrics
trainable_variables
regularization_losses
 
\Z
VARIABLE_VALUEdense_100/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_100/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

	0

1

	0

1
 
н
layer_regularization_losses
	variables
metrics

 layers
!non_trainable_variables
trainable_variables
regularization_losses
"layer_metrics
\Z
VARIABLE_VALUEdense_101/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_101/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
н
#layer_regularization_losses
	variables
$metrics

%layers
&non_trainable_variables
trainable_variables
regularization_losses
'layer_metrics
 
 
 
н
(layer_regularization_losses
	variables
)metrics

*layers
+non_trainable_variables
trainable_variables
regularization_losses
,layer_metrics
 
 

0
1
2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
В
serving_default_dense_100_inputPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
ш
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_100_inputdense_100/kerneldense_100/biasdense_101/kerneldense_101/bias*
Tin	
2*
Tout
2*'
_output_shapes
:         *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*1
f,R*
(__inference_signature_wrapper_6489832179
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Х
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_100/kernel/Read/ReadVariableOp"dense_100/bias/Read/ReadVariableOp$dense_101/kernel/Read/ReadVariableOp"dense_101/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference__traced_save_6489832329
└
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_100/kerneldense_100/biasdense_101/kerneldense_101/bias*
Tin	
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*/
f*R(
&__inference__traced_restore_6489832353√Ы
Е
е
2__inference_sequential_50_layer_call_fn_6489832241

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:         *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_sequential_50_layer_call_and_return_conditional_losses_64898321532
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ъ
▒
I__inference_dense_100_layer_call_and_return_conditional_losses_6489832252

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         d2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
└
f
J__inference_softmax_50_layer_call_and_return_conditional_losses_6489832285

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:         2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
а
о
2__inference_sequential_50_layer_call_fn_6489832164
dense_100_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCalldense_100_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:         *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_sequential_50_layer_call_and_return_conditional_losses_64898321532
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:         
)
_user_specified_namedense_100_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
О
▒
I__inference_dense_101_layer_call_and_return_conditional_losses_6489832271

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d:::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
№
K
/__inference_softmax_50_layer_call_fn_6489832290

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_softmax_50_layer_call_and_return_conditional_losses_64898320832
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Г
ж
M__inference_sequential_50_layer_call_and_return_conditional_losses_6489832197

inputs,
(dense_100_matmul_readvariableop_resource-
)dense_100_biasadd_readvariableop_resource,
(dense_101_matmul_readvariableop_resource-
)dense_101_biasadd_readvariableop_resource
identityИл
dense_100/MatMul/ReadVariableOpReadVariableOp(dense_100_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02!
dense_100/MatMul/ReadVariableOpС
dense_100/MatMulMatMulinputs'dense_100/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense_100/MatMulк
 dense_100/BiasAdd/ReadVariableOpReadVariableOp)dense_100_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02"
 dense_100/BiasAdd/ReadVariableOpй
dense_100/BiasAddBiasAdddense_100/MatMul:product:0(dense_100/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense_100/BiasAddv
dense_100/ReluReludense_100/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
dense_100/Reluл
dense_101/MatMul/ReadVariableOpReadVariableOp(dense_101_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02!
dense_101/MatMul/ReadVariableOpз
dense_101/MatMulMatMuldense_100/Relu:activations:0'dense_101/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_101/MatMulк
 dense_101/BiasAdd/ReadVariableOpReadVariableOp)dense_101_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_101/BiasAdd/ReadVariableOpй
dense_101/BiasAddBiasAdddense_101/MatMul:product:0(dense_101/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_101/BiasAddБ
softmax_50/SoftmaxSoftmaxdense_101/BiasAdd:output:0*
T0*'
_output_shapes
:         2
softmax_50/Softmaxp
IdentityIdentitysoftmax_50/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         :::::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
О
▒
I__inference_dense_101_layer_call_and_return_conditional_losses_6489832062

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d:::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
┐
е
M__inference_sequential_50_layer_call_and_return_conditional_losses_6489832092
dense_100_input
dense_100_6489832047
dense_100_6489832049
dense_101_6489832073
dense_101_6489832075
identityИв!dense_100/StatefulPartitionedCallв!dense_101/StatefulPartitionedCallМ
!dense_100/StatefulPartitionedCallStatefulPartitionedCalldense_100_inputdense_100_6489832047dense_100_6489832049*
Tin
2*
Tout
2*'
_output_shapes
:         d*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_dense_100_layer_call_and_return_conditional_losses_64898320362#
!dense_100/StatefulPartitionedCallз
!dense_101/StatefulPartitionedCallStatefulPartitionedCall*dense_100/StatefulPartitionedCall:output:0dense_101_6489832073dense_101_6489832075*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_dense_101_layer_call_and_return_conditional_losses_64898320622#
!dense_101/StatefulPartitionedCallр
softmax_50/PartitionedCallPartitionedCall*dense_101/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_softmax_50_layer_call_and_return_conditional_losses_64898320832
softmax_50/PartitionedCall┐
IdentityIdentity#softmax_50/PartitionedCall:output:0"^dense_100/StatefulPartitionedCall"^dense_101/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2F
!dense_100/StatefulPartitionedCall!dense_100/StatefulPartitionedCall2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall:X T
'
_output_shapes
:         
)
_user_specified_namedense_100_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ю
д
(__inference_signature_wrapper_6489832179
dense_100_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCalldense_100_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:         *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*.
f)R'
%__inference__wrapped_model_64898320212
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:         
)
_user_specified_namedense_100_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
д
Ь
M__inference_sequential_50_layer_call_and_return_conditional_losses_6489832153

inputs
dense_100_6489832141
dense_100_6489832143
dense_101_6489832146
dense_101_6489832148
identityИв!dense_100/StatefulPartitionedCallв!dense_101/StatefulPartitionedCallГ
!dense_100/StatefulPartitionedCallStatefulPartitionedCallinputsdense_100_6489832141dense_100_6489832143*
Tin
2*
Tout
2*'
_output_shapes
:         d*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_dense_100_layer_call_and_return_conditional_losses_64898320362#
!dense_100/StatefulPartitionedCallз
!dense_101/StatefulPartitionedCallStatefulPartitionedCall*dense_100/StatefulPartitionedCall:output:0dense_101_6489832146dense_101_6489832148*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_dense_101_layer_call_and_return_conditional_losses_64898320622#
!dense_101/StatefulPartitionedCallр
softmax_50/PartitionedCallPartitionedCall*dense_101/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_softmax_50_layer_call_and_return_conditional_losses_64898320832
softmax_50/PartitionedCall┐
IdentityIdentity#softmax_50/PartitionedCall:output:0"^dense_100/StatefulPartitionedCall"^dense_101/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2F
!dense_100/StatefulPartitionedCall!dense_100/StatefulPartitionedCall2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Б
Г
.__inference_dense_101_layer_call_fn_6489832280

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╫
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_dense_101_layer_call_and_return_conditional_losses_64898320622
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Л
┐
%__inference__wrapped_model_6489832021
dense_100_input:
6sequential_50_dense_100_matmul_readvariableop_resource;
7sequential_50_dense_100_biasadd_readvariableop_resource:
6sequential_50_dense_101_matmul_readvariableop_resource;
7sequential_50_dense_101_biasadd_readvariableop_resource
identityИ╒
-sequential_50/dense_100/MatMul/ReadVariableOpReadVariableOp6sequential_50_dense_100_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02/
-sequential_50/dense_100/MatMul/ReadVariableOp─
sequential_50/dense_100/MatMulMatMuldense_100_input5sequential_50/dense_100/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2 
sequential_50/dense_100/MatMul╘
.sequential_50/dense_100/BiasAdd/ReadVariableOpReadVariableOp7sequential_50_dense_100_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.sequential_50/dense_100/BiasAdd/ReadVariableOpс
sequential_50/dense_100/BiasAddBiasAdd(sequential_50/dense_100/MatMul:product:06sequential_50/dense_100/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2!
sequential_50/dense_100/BiasAddа
sequential_50/dense_100/ReluRelu(sequential_50/dense_100/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
sequential_50/dense_100/Relu╒
-sequential_50/dense_101/MatMul/ReadVariableOpReadVariableOp6sequential_50_dense_101_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02/
-sequential_50/dense_101/MatMul/ReadVariableOp▀
sequential_50/dense_101/MatMulMatMul*sequential_50/dense_100/Relu:activations:05sequential_50/dense_101/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2 
sequential_50/dense_101/MatMul╘
.sequential_50/dense_101/BiasAdd/ReadVariableOpReadVariableOp7sequential_50_dense_101_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_50/dense_101/BiasAdd/ReadVariableOpс
sequential_50/dense_101/BiasAddBiasAdd(sequential_50/dense_101/MatMul:product:06sequential_50/dense_101/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2!
sequential_50/dense_101/BiasAddл
 sequential_50/softmax_50/SoftmaxSoftmax(sequential_50/dense_101/BiasAdd:output:0*
T0*'
_output_shapes
:         2"
 sequential_50/softmax_50/Softmax~
IdentityIdentity*sequential_50/softmax_50/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         :::::X T
'
_output_shapes
:         
)
_user_specified_namedense_100_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Б
Г
.__inference_dense_100_layer_call_fn_6489832261

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╫
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         d*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_dense_100_layer_call_and_return_conditional_losses_64898320362
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
┐
е
M__inference_sequential_50_layer_call_and_return_conditional_losses_6489832107
dense_100_input
dense_100_6489832095
dense_100_6489832097
dense_101_6489832100
dense_101_6489832102
identityИв!dense_100/StatefulPartitionedCallв!dense_101/StatefulPartitionedCallМ
!dense_100/StatefulPartitionedCallStatefulPartitionedCalldense_100_inputdense_100_6489832095dense_100_6489832097*
Tin
2*
Tout
2*'
_output_shapes
:         d*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_dense_100_layer_call_and_return_conditional_losses_64898320362#
!dense_100/StatefulPartitionedCallз
!dense_101/StatefulPartitionedCallStatefulPartitionedCall*dense_100/StatefulPartitionedCall:output:0dense_101_6489832100dense_101_6489832102*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_dense_101_layer_call_and_return_conditional_losses_64898320622#
!dense_101/StatefulPartitionedCallр
softmax_50/PartitionedCallPartitionedCall*dense_101/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_softmax_50_layer_call_and_return_conditional_losses_64898320832
softmax_50/PartitionedCall┐
IdentityIdentity#softmax_50/PartitionedCall:output:0"^dense_100/StatefulPartitionedCall"^dense_101/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2F
!dense_100/StatefulPartitionedCall!dense_100/StatefulPartitionedCall2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall:X T
'
_output_shapes
:         
)
_user_specified_namedense_100_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ъ
▒
I__inference_dense_100_layer_call_and_return_conditional_losses_6489832036

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         d2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Э
╓
&__inference__traced_restore_6489832353
file_prefix%
!assignvariableop_dense_100_kernel%
!assignvariableop_1_dense_100_bias'
#assignvariableop_2_dense_101_kernel%
!assignvariableop_3_dense_101_bias

identity_5ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3в	RestoreV2вRestoreV2_1х
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ё
valueчBфB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesЦ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
RestoreV2/shape_and_slices┐
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityС
AssignVariableOpAssignVariableOp!assignvariableop_dense_100_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ч
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_100_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Щ
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_101_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Ч
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_101_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3и
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp║

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4╞

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Г
ж
M__inference_sequential_50_layer_call_and_return_conditional_losses_6489832215

inputs,
(dense_100_matmul_readvariableop_resource-
)dense_100_biasadd_readvariableop_resource,
(dense_101_matmul_readvariableop_resource-
)dense_101_biasadd_readvariableop_resource
identityИл
dense_100/MatMul/ReadVariableOpReadVariableOp(dense_100_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02!
dense_100/MatMul/ReadVariableOpС
dense_100/MatMulMatMulinputs'dense_100/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense_100/MatMulк
 dense_100/BiasAdd/ReadVariableOpReadVariableOp)dense_100_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02"
 dense_100/BiasAdd/ReadVariableOpй
dense_100/BiasAddBiasAdddense_100/MatMul:product:0(dense_100/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense_100/BiasAddv
dense_100/ReluReludense_100/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
dense_100/Reluл
dense_101/MatMul/ReadVariableOpReadVariableOp(dense_101_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02!
dense_101/MatMul/ReadVariableOpз
dense_101/MatMulMatMuldense_100/Relu:activations:0'dense_101/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_101/MatMulк
 dense_101/BiasAdd/ReadVariableOpReadVariableOp)dense_101_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_101/BiasAdd/ReadVariableOpй
dense_101/BiasAddBiasAdddense_101/MatMul:product:0(dense_101/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_101/BiasAddБ
softmax_50/SoftmaxSoftmaxdense_101/BiasAdd:output:0*
T0*'
_output_shapes
:         2
softmax_50/Softmaxp
IdentityIdentitysoftmax_50/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         :::::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Е
е
2__inference_sequential_50_layer_call_fn_6489832228

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:         *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_sequential_50_layer_call_and_return_conditional_losses_64898321252
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
а
о
2__inference_sequential_50_layer_call_fn_6489832136
dense_100_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCalldense_100_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:         *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_sequential_50_layer_call_and_return_conditional_losses_64898321252
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:         
)
_user_specified_namedense_100_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
└
f
J__inference_softmax_50_layer_call_and_return_conditional_losses_6489832083

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:         2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ь 
╞
#__inference__traced_save_6489832329
file_prefix/
+savev2_dense_100_kernel_read_readvariableop-
)savev2_dense_100_bias_read_readvariableop/
+savev2_dense_101_kernel_read_readvariableop-
)savev2_dense_101_bias_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1П
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_c1050017187a44f8bbf1e12424fc5912/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename▀
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ё
valueчBфB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesР
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
SaveV2/shape_and_slices▀
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_100_kernel_read_readvariableop)savev2_dense_100_bias_read_readvariableop+savev2_dense_101_kernel_read_readvariableop)savev2_dense_101_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardм
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1в
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices╧
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesм
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*7
_input_shapes&
$: :d:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: 
д
Ь
M__inference_sequential_50_layer_call_and_return_conditional_losses_6489832125

inputs
dense_100_6489832113
dense_100_6489832115
dense_101_6489832118
dense_101_6489832120
identityИв!dense_100/StatefulPartitionedCallв!dense_101/StatefulPartitionedCallГ
!dense_100/StatefulPartitionedCallStatefulPartitionedCallinputsdense_100_6489832113dense_100_6489832115*
Tin
2*
Tout
2*'
_output_shapes
:         d*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_dense_100_layer_call_and_return_conditional_losses_64898320362#
!dense_100/StatefulPartitionedCallз
!dense_101/StatefulPartitionedCallStatefulPartitionedCall*dense_100/StatefulPartitionedCall:output:0dense_101_6489832118dense_101_6489832120*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_dense_101_layer_call_and_return_conditional_losses_64898320622#
!dense_101/StatefulPartitionedCallр
softmax_50/PartitionedCallPartitionedCall*dense_101/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_softmax_50_layer_call_and_return_conditional_losses_64898320832
softmax_50/PartitionedCall┐
IdentityIdentity#softmax_50/PartitionedCall:output:0"^dense_100/StatefulPartitionedCall"^dense_101/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2F
!dense_100/StatefulPartitionedCall!dense_100/StatefulPartitionedCall2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "пL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╜
serving_defaultй
K
dense_100_input8
!serving_default_dense_100_input:0         >

softmax_500
StatefulPartitionedCall:0         tensorflow/serving/predict:еj
Ё
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api

signatures
*-&call_and_return_all_conditional_losses
.__call__
/_default_save_signature"┘
_tf_keras_sequential║{"class_name": "Sequential", "name": "sequential_50", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_50", "layers": [{"class_name": "Dense", "config": {"name": "dense_100", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_101", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Softmax", "config": {"name": "softmax_50", "trainable": true, "dtype": "float32", "axis": -1}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_50", "layers": [{"class_name": "Dense", "config": {"name": "dense_100", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_101", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Softmax", "config": {"name": "softmax_50", "trainable": true, "dtype": "float32", "axis": -1}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}}}
р

	kernel

bias
	variables
trainable_variables
regularization_losses
	keras_api
*0&call_and_return_all_conditional_losses
1__call__"╗
_tf_keras_layerб{"class_name": "Dense", "name": "dense_100", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "stateful": false, "config": {"name": "dense_100", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
╘

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*2&call_and_return_all_conditional_losses
3__call__"п
_tf_keras_layerХ{"class_name": "Dense", "name": "dense_101", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_101", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
б
	variables
trainable_variables
regularization_losses
	keras_api
*4&call_and_return_all_conditional_losses
5__call__"Т
_tf_keras_layer°{"class_name": "Softmax", "name": "softmax_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "softmax_50", "trainable": true, "dtype": "float32", "axis": -1}}
<
	0

1
2
3"
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
layer_regularization_losses
	variables
metrics

layers
non_trainable_variables
layer_metrics
trainable_variables
regularization_losses
.__call__
/_default_save_signature
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
,
6serving_default"
signature_map
": d2dense_100/kernel
:d2dense_100/bias
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
layer_regularization_losses
	variables
metrics

 layers
!non_trainable_variables
trainable_variables
regularization_losses
"layer_metrics
1__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
": d2dense_101/kernel
:2dense_101/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
#layer_regularization_losses
	variables
$metrics

%layers
&non_trainable_variables
trainable_variables
regularization_losses
'layer_metrics
3__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
(layer_regularization_losses
	variables
)metrics

*layers
+non_trainable_variables
trainable_variables
regularization_losses
,layer_metrics
5__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
В2 
M__inference_sequential_50_layer_call_and_return_conditional_losses_6489832092
M__inference_sequential_50_layer_call_and_return_conditional_losses_6489832197
M__inference_sequential_50_layer_call_and_return_conditional_losses_6489832107
M__inference_sequential_50_layer_call_and_return_conditional_losses_6489832215└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ц2У
2__inference_sequential_50_layer_call_fn_6489832164
2__inference_sequential_50_layer_call_fn_6489832228
2__inference_sequential_50_layer_call_fn_6489832136
2__inference_sequential_50_layer_call_fn_6489832241└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ы2ш
%__inference__wrapped_model_6489832021╛
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *.в+
)К&
dense_100_input         
є2Ё
I__inference_dense_100_layer_call_and_return_conditional_losses_6489832252в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪2╒
.__inference_dense_100_layer_call_fn_6489832261в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
є2Ё
I__inference_dense_101_layer_call_and_return_conditional_losses_6489832271в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪2╒
.__inference_dense_101_layer_call_fn_6489832280в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ї2ё
J__inference_softmax_50_layer_call_and_return_conditional_losses_6489832285в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┘2╓
/__inference_softmax_50_layer_call_fn_6489832290в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
?B=
(__inference_signature_wrapper_6489832179dense_100_inputв
%__inference__wrapped_model_6489832021y	
8в5
.в+
)К&
dense_100_input         
к "7к4
2

softmax_50$К!

softmax_50         й
I__inference_dense_100_layer_call_and_return_conditional_losses_6489832252\	
/в,
%в"
 К
inputs         
к "%в"
К
0         d
Ъ Б
.__inference_dense_100_layer_call_fn_6489832261O	
/в,
%в"
 К
inputs         
к "К         dй
I__inference_dense_101_layer_call_and_return_conditional_losses_6489832271\/в,
%в"
 К
inputs         d
к "%в"
К
0         
Ъ Б
.__inference_dense_101_layer_call_fn_6489832280O/в,
%в"
 К
inputs         d
к "К         └
M__inference_sequential_50_layer_call_and_return_conditional_losses_6489832092o	
@в=
6в3
)К&
dense_100_input         
p

 
к "%в"
К
0         
Ъ └
M__inference_sequential_50_layer_call_and_return_conditional_losses_6489832107o	
@в=
6в3
)К&
dense_100_input         
p 

 
к "%в"
К
0         
Ъ ╖
M__inference_sequential_50_layer_call_and_return_conditional_losses_6489832197f	
7в4
-в*
 К
inputs         
p

 
к "%в"
К
0         
Ъ ╖
M__inference_sequential_50_layer_call_and_return_conditional_losses_6489832215f	
7в4
-в*
 К
inputs         
p 

 
к "%в"
К
0         
Ъ Ш
2__inference_sequential_50_layer_call_fn_6489832136b	
@в=
6в3
)К&
dense_100_input         
p

 
к "К         Ш
2__inference_sequential_50_layer_call_fn_6489832164b	
@в=
6в3
)К&
dense_100_input         
p 

 
к "К         П
2__inference_sequential_50_layer_call_fn_6489832228Y	
7в4
-в*
 К
inputs         
p

 
к "К         П
2__inference_sequential_50_layer_call_fn_6489832241Y	
7в4
-в*
 К
inputs         
p 

 
к "К         ╣
(__inference_signature_wrapper_6489832179М	
KвH
в 
Aк>
<
dense_100_input)К&
dense_100_input         "7к4
2

softmax_50$К!

softmax_50         ж
J__inference_softmax_50_layer_call_and_return_conditional_losses_6489832285X/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ ~
/__inference_softmax_50_layer_call_fn_6489832290K/в,
%в"
 К
inputs         
к "К         