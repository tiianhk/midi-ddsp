��%
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
3
Square
x"T
y"T"
Ttype:
2
	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28��!
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:@*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:@*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:@*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:@*
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
�
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
�
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
�
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
�
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_3/gamma
�
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_3/beta
�
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:@*
dtype0
�
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_3/moving_mean
�
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
�
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_3/moving_variance
�
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/m
�
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:@*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:@*
dtype0
�
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/batch_normalization/gamma/m
�
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:@*
dtype0
�
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/batch_normalization/beta/m
�
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:@*
dtype0
�
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_1/kernel/m
�
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/m
�
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:@*
dtype0
�
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/m
�
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:@*
dtype0
�
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_2/kernel/m
�
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_2/gamma/m
�
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
:@*
dtype0
�
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_2/beta/m
�
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
:@*
dtype0
�
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_3/kernel/m
�
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_3/gamma/m
�
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes
:@*
dtype0
�
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_3/beta/m
�
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes
:@*
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	�@*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:@*
dtype0
�
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/v
�
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:@*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:@*
dtype0
�
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/batch_normalization/gamma/v
�
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:@*
dtype0
�
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/batch_normalization/beta/v
�
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:@*
dtype0
�
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_1/kernel/v
�
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/v
�
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:@*
dtype0
�
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/v
�
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:@*
dtype0
�
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_2/kernel/v
�
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_2/gamma/v
�
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
:@*
dtype0
�
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_2/beta/v
�
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
:@*
dtype0
�
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_3/kernel/v
�
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_3/gamma/v
�
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes
:@*
dtype0
�
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_3/beta/v
�
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes
:@*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	�@*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value�B� Bق
�
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer-5
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
 
 
 
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
layer-8
layer_with_weights-4
layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer-16
layer-17
 layer-18
!layer_with_weights-8
!layer-19
"	variables
#trainable_variables
$regularization_losses
%	keras_api
R
&	variables
'trainable_variables
(regularization_losses
)	keras_api
R
*	variables
+trainable_variables
,regularization_losses
-	keras_api
R
.	variables
/trainable_variables
0regularization_losses
1	keras_api
�
2iter

3beta_1

4beta_2
	5decay
6learning_rate7m�8m�9m�:m�=m�>m�?m�@m�Cm�Dm�Em�Fm�Im�Jm�Km�Lm�Om�Pm�7v�8v�9v�:v�=v�>v�?v�@v�Cv�Dv�Ev�Fv�Iv�Jv�Kv�Lv�Ov�Pv�
�
70
81
92
:3
;4
<5
=6
>7
?8
@9
A10
B11
C12
D13
E14
F15
G16
H17
I18
J19
K20
L21
M22
N23
O24
P25
�
70
81
92
:3
=4
>5
?6
@7
C8
D9
E10
F11
I12
J13
K14
L15
O16
P17
 
�
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
		variables

trainable_variables
regularization_losses
 
 
h

7kernel
8bias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
�
Zaxis
	9gamma
:beta
;moving_mean
<moving_variance
[	variables
\trainable_variables
]regularization_losses
^	keras_api
R
_	variables
`trainable_variables
aregularization_losses
b	keras_api
R
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
h

=kernel
>bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
�
kaxis
	?gamma
@beta
Amoving_mean
Bmoving_variance
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
R
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
R
t	variables
utrainable_variables
vregularization_losses
w	keras_api
h

Ckernel
Dbias
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
�
|axis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance
}	variables
~trainable_variables
regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
l

Ikernel
Jbias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�
	�axis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
l

Okernel
Pbias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�
70
81
92
:3
;4
<5
=6
>7
?8
@9
A10
B11
C12
D13
E14
F15
G16
H17
I18
J19
K20
L21
M22
N23
O24
P25
�
70
81
92
:3
=4
>5
?6
@7
C8
D9
E10
F11
I12
J13
K14
L15
O16
P17
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
&	variables
'trainable_variables
(regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEbatch_normalization/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEbatch_normalization/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_2/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_2/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_2/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_2/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_2/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_2/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_3/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_3/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_3/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_3/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUE
dense/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
8
;0
<1
A2
B3
G4
H5
M6
N7
1
0
1
2
3
4
5
6

�0
 
 

70
81

70
81
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
 

90
:1
;2
<3

90
:1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses

=0
>1

=0
>1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
 

?0
@1
A2
B3

?0
@1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses

C0
D1

C0
D1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
 

E0
F1
G2
H3

E0
F1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses

I0
J1

I0
J1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 

K0
L1
M2
N3

K0
L1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses

O0
P1

O0
P1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
8
;0
<1
A2
B3
G4
H5
M6
N7
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
 18
!19
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
 
8

�total

�count
�	variables
�	keras_api
 
 
 
 
 

;0
<1
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
 
 

A0
B1
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
 
 

G0
H1
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
 
 

M0
N1
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
lj
VARIABLE_VALUEAdam/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/batch_normalization/gamma/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/batch_normalization/beta/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_1/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_1/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_2/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_2/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/batch_normalization_2/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_3/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_3/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/batch_normalization_3/beta/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/dense/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/batch_normalization/gamma/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/batch_normalization/beta/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_1/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_1/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_2/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_2/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/batch_normalization_2/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_3/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_3/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/dense/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_anchor_inputPlaceholder*0
_output_shapes
:����������~*
dtype0*%
shape:����������~
�
serving_default_negative_inputPlaceholder*0
_output_shapes
:����������~*
dtype0*%
shape:����������~
�
serving_default_positive_inputPlaceholder*0
_output_shapes
:����������~*
dtype0*%
shape:����������~
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_anchor_inputserving_default_negative_inputserving_default_positive_inputconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense/kernel
dense/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_1185911
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*R
TinK
I2G	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_save_1187974
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense/kernel
dense/biastotalcountAdam/conv2d/kernel/mAdam/conv2d/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/dense/kernel/mAdam/dense/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/dense/kernel/vAdam/dense/bias/v*Q
TinJ
H2F*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__traced_restore_1188191��
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1183847

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�U
�
E__inference_backbone_layer_call_and_return_conditional_losses_1184327

inputs(
conv2d_1184083:@
conv2d_1184085:@)
batch_normalization_1184106:@)
batch_normalization_1184108:@)
batch_normalization_1184110:@)
batch_normalization_1184112:@*
conv2d_1_1184139:@@
conv2d_1_1184141:@+
batch_normalization_1_1184162:@+
batch_normalization_1_1184164:@+
batch_normalization_1_1184166:@+
batch_normalization_1_1184168:@*
conv2d_2_1184195:@@
conv2d_2_1184197:@+
batch_normalization_2_1184218:@+
batch_normalization_2_1184220:@+
batch_normalization_2_1184222:@+
batch_normalization_2_1184224:@*
conv2d_3_1184251:@@
conv2d_3_1184253:@+
batch_normalization_3_1184274:@+
batch_normalization_3_1184276:@+
batch_normalization_3_1184278:@+
batch_normalization_3_1184280:@ 
dense_1184321:	�@
dense_1184323:@
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�dense/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1184083conv2d_1184085*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������~@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1184082�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1184106batch_normalization_1184108batch_normalization_1184110batch_normalization_1184112*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������~@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1184105�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������~@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1184120�
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1184126�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_1184139conv2d_1_1184141*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1184138�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_1184162batch_normalization_1_1184164batch_normalization_1_1184166batch_normalization_1_1184168*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1184161�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1184176�
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1184182�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_1184195conv2d_2_1184197*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1184194�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_1184218batch_normalization_2_1184220batch_normalization_2_1184222batch_normalization_2_1184224*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1184217�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_1184232�
max_pooling2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1184238�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_1184251conv2d_3_1184253*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1184250�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_1184274batch_normalization_3_1184276batch_normalization_3_1184278batch_normalization_3_1184280*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1184273�
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_1184288�
max_pooling2d_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1184294�
max_pooling2d_4/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1184300�
flatten/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1184308�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1184321dense_1184323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1184320u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:X T
0
_output_shapes
:����������~
 
_user_specified_nameinputs
�
E
)__inference_flatten_layer_call_fn_1187717

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1184308a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_2_layer_call_fn_1187417

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1184512w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1187626

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_2_layer_call_fn_1187391

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1183954�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_1185911
anchor_input
negative_input
positive_input!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	�@

unknown_24:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallanchor_inputpositive_inputnegative_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_1183749o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������~:����������~:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:����������~
&
_user_specified_nameanchor_input:`\
0
_output_shapes
:����������~
(
_user_specified_namenegative_input:`\
0
_output_shapes
:����������~
(
_user_specified_namepositive_input
�	
�
5__inference_batch_normalization_layer_call_fn_1187032

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1183771�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
K
/__inference_max_pooling2d_layer_call_fn_1187163

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1184126h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@?@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������~@:X T
0
_output_shapes
:����������~@
 
_user_specified_nameinputs
�

�
C__inference_conv2d_layer_call_and_return_conditional_losses_1184082

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:����������~@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������~: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������~
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1187280

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1187687

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_4_layer_call_fn_1187697

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1184062�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_1_layer_call_fn_1187218

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1183878�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�.
�
D__inference_triplet_layer_call_and_return_conditional_losses_1185844
anchor_input
positive_input
negative_input*
backbone_1185733:@
backbone_1185735:@
backbone_1185737:@
backbone_1185739:@
backbone_1185741:@
backbone_1185743:@*
backbone_1185745:@@
backbone_1185747:@
backbone_1185749:@
backbone_1185751:@
backbone_1185753:@
backbone_1185755:@*
backbone_1185757:@@
backbone_1185759:@
backbone_1185761:@
backbone_1185763:@
backbone_1185765:@
backbone_1185767:@*
backbone_1185769:@@
backbone_1185771:@
backbone_1185773:@
backbone_1185775:@
backbone_1185777:@
backbone_1185779:@#
backbone_1185781:	�@
backbone_1185783:@
identity�� backbone/StatefulPartitionedCall�"backbone/StatefulPartitionedCall_1�"backbone/StatefulPartitionedCall_2�
 backbone/StatefulPartitionedCallStatefulPartitionedCallanchor_inputbackbone_1185733backbone_1185735backbone_1185737backbone_1185739backbone_1185741backbone_1185743backbone_1185745backbone_1185747backbone_1185749backbone_1185751backbone_1185753backbone_1185755backbone_1185757backbone_1185759backbone_1185761backbone_1185763backbone_1185765backbone_1185767backbone_1185769backbone_1185771backbone_1185773backbone_1185775backbone_1185777backbone_1185779backbone_1185781backbone_1185783*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_backbone_layer_call_and_return_conditional_losses_1184797�
"backbone/StatefulPartitionedCall_1StatefulPartitionedCallnegative_inputbackbone_1185733backbone_1185735backbone_1185737backbone_1185739backbone_1185741backbone_1185743backbone_1185745backbone_1185747backbone_1185749backbone_1185751backbone_1185753backbone_1185755backbone_1185757backbone_1185759backbone_1185761backbone_1185763backbone_1185765backbone_1185767backbone_1185769backbone_1185771backbone_1185773backbone_1185775backbone_1185777backbone_1185779backbone_1185781backbone_1185783!^backbone/StatefulPartitionedCall*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_backbone_layer_call_and_return_conditional_losses_1184797�
"backbone/StatefulPartitionedCall_2StatefulPartitionedCallpositive_inputbackbone_1185733backbone_1185735backbone_1185737backbone_1185739backbone_1185741backbone_1185743backbone_1185745backbone_1185747backbone_1185749backbone_1185751backbone_1185753backbone_1185755backbone_1185757backbone_1185759backbone_1185761backbone_1185763backbone_1185765backbone_1185767backbone_1185769backbone_1185771backbone_1185773backbone_1185775backbone_1185777backbone_1185779backbone_1185781backbone_1185783#^backbone/StatefulPartitionedCall_1*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_backbone_layer_call_and_return_conditional_losses_1184797�
dot/PartitionedCallPartitionedCall)backbone/StatefulPartitionedCall:output:0+backbone/StatefulPartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dot_layer_call_and_return_conditional_losses_1185203�
dot_1/PartitionedCallPartitionedCall)backbone/StatefulPartitionedCall:output:0+backbone/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dot_1_layer_call_and_return_conditional_losses_1185231�
concatenate/PartitionedCallPartitionedCalldot/PartitionedCall:output:0dot_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1185240s
IdentityIdentity$concatenate/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^backbone/StatefulPartitionedCall#^backbone/StatefulPartitionedCall_1#^backbone/StatefulPartitionedCall_2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������~:����������~:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 backbone/StatefulPartitionedCall backbone/StatefulPartitionedCall2H
"backbone/StatefulPartitionedCall_1"backbone/StatefulPartitionedCall_12H
"backbone/StatefulPartitionedCall_2"backbone/StatefulPartitionedCall_2:^ Z
0
_output_shapes
:����������~
&
_user_specified_nameanchor_input:`\
0
_output_shapes
:����������~
(
_user_specified_namepositive_input:`\
0
_output_shapes
:����������~
(
_user_specified_namenegative_input
�
h
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1183974

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_1_layer_call_fn_1187331

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1183898�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_3_layer_call_fn_1187564

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1184030�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1187316

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@?@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������@?@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@?@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������@?@
 
_user_specified_nameinputs
�
r
H__inference_concatenate_layer_call_and_return_conditional_losses_1185240

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_2_layer_call_fn_1187504

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1183974�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1187365

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:��������� @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
��
�6
"__inference__wrapped_model_1183749
anchor_input
positive_input
negative_inputP
6triplet_backbone_conv2d_conv2d_readvariableop_resource:@E
7triplet_backbone_conv2d_biasadd_readvariableop_resource:@J
<triplet_backbone_batch_normalization_readvariableop_resource:@L
>triplet_backbone_batch_normalization_readvariableop_1_resource:@[
Mtriplet_backbone_batch_normalization_fusedbatchnormv3_readvariableop_resource:@]
Otriplet_backbone_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@R
8triplet_backbone_conv2d_1_conv2d_readvariableop_resource:@@G
9triplet_backbone_conv2d_1_biasadd_readvariableop_resource:@L
>triplet_backbone_batch_normalization_1_readvariableop_resource:@N
@triplet_backbone_batch_normalization_1_readvariableop_1_resource:@]
Otriplet_backbone_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@_
Qtriplet_backbone_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@R
8triplet_backbone_conv2d_2_conv2d_readvariableop_resource:@@G
9triplet_backbone_conv2d_2_biasadd_readvariableop_resource:@L
>triplet_backbone_batch_normalization_2_readvariableop_resource:@N
@triplet_backbone_batch_normalization_2_readvariableop_1_resource:@]
Otriplet_backbone_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@_
Qtriplet_backbone_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@R
8triplet_backbone_conv2d_3_conv2d_readvariableop_resource:@@G
9triplet_backbone_conv2d_3_biasadd_readvariableop_resource:@L
>triplet_backbone_batch_normalization_3_readvariableop_resource:@N
@triplet_backbone_batch_normalization_3_readvariableop_1_resource:@]
Otriplet_backbone_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:@_
Qtriplet_backbone_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:@H
5triplet_backbone_dense_matmul_readvariableop_resource:	�@D
6triplet_backbone_dense_biasadd_readvariableop_resource:@
identity��Dtriplet/backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp�Ftriplet/backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�Ftriplet/backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp�Htriplet/backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1�Ftriplet/backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp�Htriplet/backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_1�3triplet/backbone/batch_normalization/ReadVariableOp�5triplet/backbone/batch_normalization/ReadVariableOp_1�5triplet/backbone/batch_normalization/ReadVariableOp_2�5triplet/backbone/batch_normalization/ReadVariableOp_3�5triplet/backbone/batch_normalization/ReadVariableOp_4�5triplet/backbone/batch_normalization/ReadVariableOp_5�Ftriplet/backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�Htriplet/backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�Htriplet/backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp�Jtriplet/backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1�Htriplet/backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp�Jtriplet/backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_1�5triplet/backbone/batch_normalization_1/ReadVariableOp�7triplet/backbone/batch_normalization_1/ReadVariableOp_1�7triplet/backbone/batch_normalization_1/ReadVariableOp_2�7triplet/backbone/batch_normalization_1/ReadVariableOp_3�7triplet/backbone/batch_normalization_1/ReadVariableOp_4�7triplet/backbone/batch_normalization_1/ReadVariableOp_5�Ftriplet/backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�Htriplet/backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�Htriplet/backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp�Jtriplet/backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1�Htriplet/backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp�Jtriplet/backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_1�5triplet/backbone/batch_normalization_2/ReadVariableOp�7triplet/backbone/batch_normalization_2/ReadVariableOp_1�7triplet/backbone/batch_normalization_2/ReadVariableOp_2�7triplet/backbone/batch_normalization_2/ReadVariableOp_3�7triplet/backbone/batch_normalization_2/ReadVariableOp_4�7triplet/backbone/batch_normalization_2/ReadVariableOp_5�Ftriplet/backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�Htriplet/backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�Htriplet/backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp�Jtriplet/backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1�Htriplet/backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp�Jtriplet/backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_1�5triplet/backbone/batch_normalization_3/ReadVariableOp�7triplet/backbone/batch_normalization_3/ReadVariableOp_1�7triplet/backbone/batch_normalization_3/ReadVariableOp_2�7triplet/backbone/batch_normalization_3/ReadVariableOp_3�7triplet/backbone/batch_normalization_3/ReadVariableOp_4�7triplet/backbone/batch_normalization_3/ReadVariableOp_5�.triplet/backbone/conv2d/BiasAdd/ReadVariableOp�0triplet/backbone/conv2d/BiasAdd_1/ReadVariableOp�0triplet/backbone/conv2d/BiasAdd_2/ReadVariableOp�-triplet/backbone/conv2d/Conv2D/ReadVariableOp�/triplet/backbone/conv2d/Conv2D_1/ReadVariableOp�/triplet/backbone/conv2d/Conv2D_2/ReadVariableOp�0triplet/backbone/conv2d_1/BiasAdd/ReadVariableOp�2triplet/backbone/conv2d_1/BiasAdd_1/ReadVariableOp�2triplet/backbone/conv2d_1/BiasAdd_2/ReadVariableOp�/triplet/backbone/conv2d_1/Conv2D/ReadVariableOp�1triplet/backbone/conv2d_1/Conv2D_1/ReadVariableOp�1triplet/backbone/conv2d_1/Conv2D_2/ReadVariableOp�0triplet/backbone/conv2d_2/BiasAdd/ReadVariableOp�2triplet/backbone/conv2d_2/BiasAdd_1/ReadVariableOp�2triplet/backbone/conv2d_2/BiasAdd_2/ReadVariableOp�/triplet/backbone/conv2d_2/Conv2D/ReadVariableOp�1triplet/backbone/conv2d_2/Conv2D_1/ReadVariableOp�1triplet/backbone/conv2d_2/Conv2D_2/ReadVariableOp�0triplet/backbone/conv2d_3/BiasAdd/ReadVariableOp�2triplet/backbone/conv2d_3/BiasAdd_1/ReadVariableOp�2triplet/backbone/conv2d_3/BiasAdd_2/ReadVariableOp�/triplet/backbone/conv2d_3/Conv2D/ReadVariableOp�1triplet/backbone/conv2d_3/Conv2D_1/ReadVariableOp�1triplet/backbone/conv2d_3/Conv2D_2/ReadVariableOp�-triplet/backbone/dense/BiasAdd/ReadVariableOp�/triplet/backbone/dense/BiasAdd_1/ReadVariableOp�/triplet/backbone/dense/BiasAdd_2/ReadVariableOp�,triplet/backbone/dense/MatMul/ReadVariableOp�.triplet/backbone/dense/MatMul_1/ReadVariableOp�.triplet/backbone/dense/MatMul_2/ReadVariableOp�
-triplet/backbone/conv2d/Conv2D/ReadVariableOpReadVariableOp6triplet_backbone_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
triplet/backbone/conv2d/Conv2DConv2Danchor_input5triplet/backbone/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@*
paddingSAME*
strides
�
.triplet/backbone/conv2d/BiasAdd/ReadVariableOpReadVariableOp7triplet_backbone_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
triplet/backbone/conv2d/BiasAddBiasAdd'triplet/backbone/conv2d/Conv2D:output:06triplet/backbone/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@�
3triplet/backbone/batch_normalization/ReadVariableOpReadVariableOp<triplet_backbone_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
5triplet/backbone/batch_normalization/ReadVariableOp_1ReadVariableOp>triplet_backbone_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Dtriplet/backbone/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpMtriplet_backbone_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Ftriplet/backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOtriplet_backbone_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
5triplet/backbone/batch_normalization/FusedBatchNormV3FusedBatchNormV3(triplet/backbone/conv2d/BiasAdd:output:0;triplet/backbone/batch_normalization/ReadVariableOp:value:0=triplet/backbone/batch_normalization/ReadVariableOp_1:value:0Ltriplet/backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Ntriplet/backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:����������~@:@:@:@:@:*
epsilon%o�:*
is_training( �
 triplet/backbone/activation/ReluRelu9triplet/backbone/batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������~@�
&triplet/backbone/max_pooling2d/MaxPoolMaxPool.triplet/backbone/activation/Relu:activations:0*/
_output_shapes
:���������@?@*
ksize
*
paddingVALID*
strides
�
/triplet/backbone/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8triplet_backbone_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
 triplet/backbone/conv2d_1/Conv2DConv2D/triplet/backbone/max_pooling2d/MaxPool:output:07triplet/backbone/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@*
paddingSAME*
strides
�
0triplet/backbone/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9triplet_backbone_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
!triplet/backbone/conv2d_1/BiasAddBiasAdd)triplet/backbone/conv2d_1/Conv2D:output:08triplet/backbone/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@�
5triplet/backbone/batch_normalization_1/ReadVariableOpReadVariableOp>triplet_backbone_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
7triplet/backbone/batch_normalization_1/ReadVariableOp_1ReadVariableOp@triplet_backbone_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Ftriplet/backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpOtriplet_backbone_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Htriplet/backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQtriplet_backbone_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
7triplet/backbone/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3*triplet/backbone/conv2d_1/BiasAdd:output:0=triplet/backbone/batch_normalization_1/ReadVariableOp:value:0?triplet/backbone/batch_normalization_1/ReadVariableOp_1:value:0Ntriplet/backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Ptriplet/backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@?@:@:@:@:@:*
epsilon%o�:*
is_training( �
"triplet/backbone/activation_1/ReluRelu;triplet/backbone/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@?@�
(triplet/backbone/max_pooling2d_1/MaxPoolMaxPool0triplet/backbone/activation_1/Relu:activations:0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
�
/triplet/backbone/conv2d_2/Conv2D/ReadVariableOpReadVariableOp8triplet_backbone_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
 triplet/backbone/conv2d_2/Conv2DConv2D1triplet/backbone/max_pooling2d_1/MaxPool:output:07triplet/backbone/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
�
0triplet/backbone/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp9triplet_backbone_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
!triplet/backbone/conv2d_2/BiasAddBiasAdd)triplet/backbone/conv2d_2/Conv2D:output:08triplet/backbone/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @�
5triplet/backbone/batch_normalization_2/ReadVariableOpReadVariableOp>triplet_backbone_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
7triplet/backbone/batch_normalization_2/ReadVariableOp_1ReadVariableOp@triplet_backbone_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Ftriplet/backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpOtriplet_backbone_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Htriplet/backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQtriplet_backbone_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
7triplet/backbone/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3*triplet/backbone/conv2d_2/BiasAdd:output:0=triplet/backbone/batch_normalization_2/ReadVariableOp:value:0?triplet/backbone/batch_normalization_2/ReadVariableOp_1:value:0Ntriplet/backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Ptriplet/backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� @:@:@:@:@:*
epsilon%o�:*
is_training( �
"triplet/backbone/activation_2/ReluRelu;triplet/backbone/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� @�
(triplet/backbone/max_pooling2d_2/MaxPoolMaxPool0triplet/backbone/activation_2/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
/triplet/backbone/conv2d_3/Conv2D/ReadVariableOpReadVariableOp8triplet_backbone_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
 triplet/backbone/conv2d_3/Conv2DConv2D1triplet/backbone/max_pooling2d_2/MaxPool:output:07triplet/backbone/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
0triplet/backbone/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp9triplet_backbone_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
!triplet/backbone/conv2d_3/BiasAddBiasAdd)triplet/backbone/conv2d_3/Conv2D:output:08triplet/backbone/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
5triplet/backbone/batch_normalization_3/ReadVariableOpReadVariableOp>triplet_backbone_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7triplet/backbone/batch_normalization_3/ReadVariableOp_1ReadVariableOp@triplet_backbone_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Ftriplet/backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpOtriplet_backbone_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Htriplet/backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQtriplet_backbone_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
7triplet/backbone/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3*triplet/backbone/conv2d_3/BiasAdd:output:0=triplet/backbone/batch_normalization_3/ReadVariableOp:value:0?triplet/backbone/batch_normalization_3/ReadVariableOp_1:value:0Ntriplet/backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Ptriplet/backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
"triplet/backbone/activation_3/ReluRelu;triplet/backbone/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@�
(triplet/backbone/max_pooling2d_3/MaxPoolMaxPool0triplet/backbone/activation_3/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
(triplet/backbone/max_pooling2d_4/MaxPoolMaxPool1triplet/backbone/max_pooling2d_3/MaxPool:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
o
triplet/backbone/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
 triplet/backbone/flatten/ReshapeReshape1triplet/backbone/max_pooling2d_4/MaxPool:output:0'triplet/backbone/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
,triplet/backbone/dense/MatMul/ReadVariableOpReadVariableOp5triplet_backbone_dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
triplet/backbone/dense/MatMulMatMul)triplet/backbone/flatten/Reshape:output:04triplet/backbone/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
-triplet/backbone/dense/BiasAdd/ReadVariableOpReadVariableOp6triplet_backbone_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
triplet/backbone/dense/BiasAddBiasAdd'triplet/backbone/dense/MatMul:product:05triplet/backbone/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/triplet/backbone/conv2d/Conv2D_1/ReadVariableOpReadVariableOp6triplet_backbone_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
 triplet/backbone/conv2d/Conv2D_1Conv2Dnegative_input7triplet/backbone/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@*
paddingSAME*
strides
�
0triplet/backbone/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp7triplet_backbone_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
!triplet/backbone/conv2d/BiasAdd_1BiasAdd)triplet/backbone/conv2d/Conv2D_1:output:08triplet/backbone/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@�
5triplet/backbone/batch_normalization/ReadVariableOp_2ReadVariableOp<triplet_backbone_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
5triplet/backbone/batch_normalization/ReadVariableOp_3ReadVariableOp>triplet_backbone_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Ftriplet/backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOpReadVariableOpMtriplet_backbone_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Htriplet/backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpOtriplet_backbone_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
7triplet/backbone/batch_normalization/FusedBatchNormV3_1FusedBatchNormV3*triplet/backbone/conv2d/BiasAdd_1:output:0=triplet/backbone/batch_normalization/ReadVariableOp_2:value:0=triplet/backbone/batch_normalization/ReadVariableOp_3:value:0Ntriplet/backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp:value:0Ptriplet/backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:����������~@:@:@:@:@:*
epsilon%o�:*
is_training( �
"triplet/backbone/activation/Relu_1Relu;triplet/backbone/batch_normalization/FusedBatchNormV3_1:y:0*
T0*0
_output_shapes
:����������~@�
(triplet/backbone/max_pooling2d/MaxPool_1MaxPool0triplet/backbone/activation/Relu_1:activations:0*/
_output_shapes
:���������@?@*
ksize
*
paddingVALID*
strides
�
1triplet/backbone/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp8triplet_backbone_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
"triplet/backbone/conv2d_1/Conv2D_1Conv2D1triplet/backbone/max_pooling2d/MaxPool_1:output:09triplet/backbone/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@*
paddingSAME*
strides
�
2triplet/backbone/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp9triplet_backbone_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
#triplet/backbone/conv2d_1/BiasAdd_1BiasAdd+triplet/backbone/conv2d_1/Conv2D_1:output:0:triplet/backbone/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@�
7triplet/backbone/batch_normalization_1/ReadVariableOp_2ReadVariableOp>triplet_backbone_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
7triplet/backbone/batch_normalization_1/ReadVariableOp_3ReadVariableOp@triplet_backbone_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Htriplet/backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpReadVariableOpOtriplet_backbone_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Jtriplet/backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpQtriplet_backbone_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
9triplet/backbone/batch_normalization_1/FusedBatchNormV3_1FusedBatchNormV3,triplet/backbone/conv2d_1/BiasAdd_1:output:0?triplet/backbone/batch_normalization_1/ReadVariableOp_2:value:0?triplet/backbone/batch_normalization_1/ReadVariableOp_3:value:0Ptriplet/backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp:value:0Rtriplet/backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@?@:@:@:@:@:*
epsilon%o�:*
is_training( �
$triplet/backbone/activation_1/Relu_1Relu=triplet/backbone/batch_normalization_1/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:���������@?@�
*triplet/backbone/max_pooling2d_1/MaxPool_1MaxPool2triplet/backbone/activation_1/Relu_1:activations:0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
�
1triplet/backbone/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOp8triplet_backbone_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
"triplet/backbone/conv2d_2/Conv2D_1Conv2D3triplet/backbone/max_pooling2d_1/MaxPool_1:output:09triplet/backbone/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
�
2triplet/backbone/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOp9triplet_backbone_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
#triplet/backbone/conv2d_2/BiasAdd_1BiasAdd+triplet/backbone/conv2d_2/Conv2D_1:output:0:triplet/backbone/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @�
7triplet/backbone/batch_normalization_2/ReadVariableOp_2ReadVariableOp>triplet_backbone_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
7triplet/backbone/batch_normalization_2/ReadVariableOp_3ReadVariableOp@triplet_backbone_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Htriplet/backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpReadVariableOpOtriplet_backbone_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Jtriplet/backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpQtriplet_backbone_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
9triplet/backbone/batch_normalization_2/FusedBatchNormV3_1FusedBatchNormV3,triplet/backbone/conv2d_2/BiasAdd_1:output:0?triplet/backbone/batch_normalization_2/ReadVariableOp_2:value:0?triplet/backbone/batch_normalization_2/ReadVariableOp_3:value:0Ptriplet/backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp:value:0Rtriplet/backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� @:@:@:@:@:*
epsilon%o�:*
is_training( �
$triplet/backbone/activation_2/Relu_1Relu=triplet/backbone/batch_normalization_2/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:��������� @�
*triplet/backbone/max_pooling2d_2/MaxPool_1MaxPool2triplet/backbone/activation_2/Relu_1:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
1triplet/backbone/conv2d_3/Conv2D_1/ReadVariableOpReadVariableOp8triplet_backbone_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
"triplet/backbone/conv2d_3/Conv2D_1Conv2D3triplet/backbone/max_pooling2d_2/MaxPool_1:output:09triplet/backbone/conv2d_3/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
2triplet/backbone/conv2d_3/BiasAdd_1/ReadVariableOpReadVariableOp9triplet_backbone_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
#triplet/backbone/conv2d_3/BiasAdd_1BiasAdd+triplet/backbone/conv2d_3/Conv2D_1:output:0:triplet/backbone/conv2d_3/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
7triplet/backbone/batch_normalization_3/ReadVariableOp_2ReadVariableOp>triplet_backbone_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7triplet/backbone/batch_normalization_3/ReadVariableOp_3ReadVariableOp@triplet_backbone_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Htriplet/backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpReadVariableOpOtriplet_backbone_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Jtriplet/backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpQtriplet_backbone_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
9triplet/backbone/batch_normalization_3/FusedBatchNormV3_1FusedBatchNormV3,triplet/backbone/conv2d_3/BiasAdd_1:output:0?triplet/backbone/batch_normalization_3/ReadVariableOp_2:value:0?triplet/backbone/batch_normalization_3/ReadVariableOp_3:value:0Ptriplet/backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp:value:0Rtriplet/backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
$triplet/backbone/activation_3/Relu_1Relu=triplet/backbone/batch_normalization_3/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:���������@�
*triplet/backbone/max_pooling2d_3/MaxPool_1MaxPool2triplet/backbone/activation_3/Relu_1:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
*triplet/backbone/max_pooling2d_4/MaxPool_1MaxPool3triplet/backbone/max_pooling2d_3/MaxPool_1:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
q
 triplet/backbone/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"����   �
"triplet/backbone/flatten/Reshape_1Reshape3triplet/backbone/max_pooling2d_4/MaxPool_1:output:0)triplet/backbone/flatten/Const_1:output:0*
T0*(
_output_shapes
:�����������
.triplet/backbone/dense/MatMul_1/ReadVariableOpReadVariableOp5triplet_backbone_dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
triplet/backbone/dense/MatMul_1MatMul+triplet/backbone/flatten/Reshape_1:output:06triplet/backbone/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/triplet/backbone/dense/BiasAdd_1/ReadVariableOpReadVariableOp6triplet_backbone_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 triplet/backbone/dense/BiasAdd_1BiasAdd)triplet/backbone/dense/MatMul_1:product:07triplet/backbone/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/triplet/backbone/conv2d/Conv2D_2/ReadVariableOpReadVariableOp6triplet_backbone_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
 triplet/backbone/conv2d/Conv2D_2Conv2Dpositive_input7triplet/backbone/conv2d/Conv2D_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@*
paddingSAME*
strides
�
0triplet/backbone/conv2d/BiasAdd_2/ReadVariableOpReadVariableOp7triplet_backbone_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
!triplet/backbone/conv2d/BiasAdd_2BiasAdd)triplet/backbone/conv2d/Conv2D_2:output:08triplet/backbone/conv2d/BiasAdd_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@�
5triplet/backbone/batch_normalization/ReadVariableOp_4ReadVariableOp<triplet_backbone_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
5triplet/backbone/batch_normalization/ReadVariableOp_5ReadVariableOp>triplet_backbone_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Ftriplet/backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOpReadVariableOpMtriplet_backbone_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Htriplet/backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_1ReadVariableOpOtriplet_backbone_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
7triplet/backbone/batch_normalization/FusedBatchNormV3_2FusedBatchNormV3*triplet/backbone/conv2d/BiasAdd_2:output:0=triplet/backbone/batch_normalization/ReadVariableOp_4:value:0=triplet/backbone/batch_normalization/ReadVariableOp_5:value:0Ntriplet/backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp:value:0Ptriplet/backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:����������~@:@:@:@:@:*
epsilon%o�:*
is_training( �
"triplet/backbone/activation/Relu_2Relu;triplet/backbone/batch_normalization/FusedBatchNormV3_2:y:0*
T0*0
_output_shapes
:����������~@�
(triplet/backbone/max_pooling2d/MaxPool_2MaxPool0triplet/backbone/activation/Relu_2:activations:0*/
_output_shapes
:���������@?@*
ksize
*
paddingVALID*
strides
�
1triplet/backbone/conv2d_1/Conv2D_2/ReadVariableOpReadVariableOp8triplet_backbone_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
"triplet/backbone/conv2d_1/Conv2D_2Conv2D1triplet/backbone/max_pooling2d/MaxPool_2:output:09triplet/backbone/conv2d_1/Conv2D_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@*
paddingSAME*
strides
�
2triplet/backbone/conv2d_1/BiasAdd_2/ReadVariableOpReadVariableOp9triplet_backbone_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
#triplet/backbone/conv2d_1/BiasAdd_2BiasAdd+triplet/backbone/conv2d_1/Conv2D_2:output:0:triplet/backbone/conv2d_1/BiasAdd_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@�
7triplet/backbone/batch_normalization_1/ReadVariableOp_4ReadVariableOp>triplet_backbone_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
7triplet/backbone/batch_normalization_1/ReadVariableOp_5ReadVariableOp@triplet_backbone_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Htriplet/backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOpReadVariableOpOtriplet_backbone_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Jtriplet/backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_1ReadVariableOpQtriplet_backbone_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
9triplet/backbone/batch_normalization_1/FusedBatchNormV3_2FusedBatchNormV3,triplet/backbone/conv2d_1/BiasAdd_2:output:0?triplet/backbone/batch_normalization_1/ReadVariableOp_4:value:0?triplet/backbone/batch_normalization_1/ReadVariableOp_5:value:0Ptriplet/backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp:value:0Rtriplet/backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@?@:@:@:@:@:*
epsilon%o�:*
is_training( �
$triplet/backbone/activation_1/Relu_2Relu=triplet/backbone/batch_normalization_1/FusedBatchNormV3_2:y:0*
T0*/
_output_shapes
:���������@?@�
*triplet/backbone/max_pooling2d_1/MaxPool_2MaxPool2triplet/backbone/activation_1/Relu_2:activations:0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
�
1triplet/backbone/conv2d_2/Conv2D_2/ReadVariableOpReadVariableOp8triplet_backbone_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
"triplet/backbone/conv2d_2/Conv2D_2Conv2D3triplet/backbone/max_pooling2d_1/MaxPool_2:output:09triplet/backbone/conv2d_2/Conv2D_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
�
2triplet/backbone/conv2d_2/BiasAdd_2/ReadVariableOpReadVariableOp9triplet_backbone_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
#triplet/backbone/conv2d_2/BiasAdd_2BiasAdd+triplet/backbone/conv2d_2/Conv2D_2:output:0:triplet/backbone/conv2d_2/BiasAdd_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @�
7triplet/backbone/batch_normalization_2/ReadVariableOp_4ReadVariableOp>triplet_backbone_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
7triplet/backbone/batch_normalization_2/ReadVariableOp_5ReadVariableOp@triplet_backbone_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Htriplet/backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOpReadVariableOpOtriplet_backbone_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Jtriplet/backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_1ReadVariableOpQtriplet_backbone_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
9triplet/backbone/batch_normalization_2/FusedBatchNormV3_2FusedBatchNormV3,triplet/backbone/conv2d_2/BiasAdd_2:output:0?triplet/backbone/batch_normalization_2/ReadVariableOp_4:value:0?triplet/backbone/batch_normalization_2/ReadVariableOp_5:value:0Ptriplet/backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp:value:0Rtriplet/backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� @:@:@:@:@:*
epsilon%o�:*
is_training( �
$triplet/backbone/activation_2/Relu_2Relu=triplet/backbone/batch_normalization_2/FusedBatchNormV3_2:y:0*
T0*/
_output_shapes
:��������� @�
*triplet/backbone/max_pooling2d_2/MaxPool_2MaxPool2triplet/backbone/activation_2/Relu_2:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
1triplet/backbone/conv2d_3/Conv2D_2/ReadVariableOpReadVariableOp8triplet_backbone_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
"triplet/backbone/conv2d_3/Conv2D_2Conv2D3triplet/backbone/max_pooling2d_2/MaxPool_2:output:09triplet/backbone/conv2d_3/Conv2D_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
2triplet/backbone/conv2d_3/BiasAdd_2/ReadVariableOpReadVariableOp9triplet_backbone_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
#triplet/backbone/conv2d_3/BiasAdd_2BiasAdd+triplet/backbone/conv2d_3/Conv2D_2:output:0:triplet/backbone/conv2d_3/BiasAdd_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
7triplet/backbone/batch_normalization_3/ReadVariableOp_4ReadVariableOp>triplet_backbone_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7triplet/backbone/batch_normalization_3/ReadVariableOp_5ReadVariableOp@triplet_backbone_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Htriplet/backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOpReadVariableOpOtriplet_backbone_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Jtriplet/backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_1ReadVariableOpQtriplet_backbone_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
9triplet/backbone/batch_normalization_3/FusedBatchNormV3_2FusedBatchNormV3,triplet/backbone/conv2d_3/BiasAdd_2:output:0?triplet/backbone/batch_normalization_3/ReadVariableOp_4:value:0?triplet/backbone/batch_normalization_3/ReadVariableOp_5:value:0Ptriplet/backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp:value:0Rtriplet/backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
$triplet/backbone/activation_3/Relu_2Relu=triplet/backbone/batch_normalization_3/FusedBatchNormV3_2:y:0*
T0*/
_output_shapes
:���������@�
*triplet/backbone/max_pooling2d_3/MaxPool_2MaxPool2triplet/backbone/activation_3/Relu_2:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
*triplet/backbone/max_pooling2d_4/MaxPool_2MaxPool3triplet/backbone/max_pooling2d_3/MaxPool_2:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
q
 triplet/backbone/flatten/Const_2Const*
_output_shapes
:*
dtype0*
valueB"����   �
"triplet/backbone/flatten/Reshape_2Reshape3triplet/backbone/max_pooling2d_4/MaxPool_2:output:0)triplet/backbone/flatten/Const_2:output:0*
T0*(
_output_shapes
:�����������
.triplet/backbone/dense/MatMul_2/ReadVariableOpReadVariableOp5triplet_backbone_dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
triplet/backbone/dense/MatMul_2MatMul+triplet/backbone/flatten/Reshape_2:output:06triplet/backbone/dense/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/triplet/backbone/dense/BiasAdd_2/ReadVariableOpReadVariableOp6triplet_backbone_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 triplet/backbone/dense/BiasAdd_2BiasAdd)triplet/backbone/dense/MatMul_2:product:07triplet/backbone/dense/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
triplet/dot/l2_normalize/SquareSquare'triplet/backbone/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@p
.triplet/dot/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
triplet/dot/l2_normalize/SumSum#triplet/dot/l2_normalize/Square:y:07triplet/dot/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(g
"triplet/dot/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
 triplet/dot/l2_normalize/MaximumMaximum%triplet/dot/l2_normalize/Sum:output:0+triplet/dot/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������
triplet/dot/l2_normalize/RsqrtRsqrt$triplet/dot/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
triplet/dot/l2_normalizeMul'triplet/backbone/dense/BiasAdd:output:0"triplet/dot/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@�
!triplet/dot/l2_normalize_1/SquareSquare)triplet/backbone/dense/BiasAdd_2:output:0*
T0*'
_output_shapes
:���������@r
0triplet/dot/l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
triplet/dot/l2_normalize_1/SumSum%triplet/dot/l2_normalize_1/Square:y:09triplet/dot/l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(i
$triplet/dot/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
"triplet/dot/l2_normalize_1/MaximumMaximum'triplet/dot/l2_normalize_1/Sum:output:0-triplet/dot/l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:����������
 triplet/dot/l2_normalize_1/RsqrtRsqrt&triplet/dot/l2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:����������
triplet/dot/l2_normalize_1Mul)triplet/backbone/dense/BiasAdd_2:output:0$triplet/dot/l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:���������@\
triplet/dot/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
triplet/dot/ExpandDims
ExpandDimstriplet/dot/l2_normalize:z:0#triplet/dot/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������@^
triplet/dot/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
triplet/dot/ExpandDims_1
ExpandDimstriplet/dot/l2_normalize_1:z:0%triplet/dot/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:���������@�
triplet/dot/MatMulBatchMatMulV2triplet/dot/ExpandDims:output:0!triplet/dot/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������\
triplet/dot/ShapeShapetriplet/dot/MatMul:output:0*
T0*
_output_shapes
:�
triplet/dot/SqueezeSqueezetriplet/dot/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
�
!triplet/dot_1/l2_normalize/SquareSquare'triplet/backbone/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@r
0triplet/dot_1/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
triplet/dot_1/l2_normalize/SumSum%triplet/dot_1/l2_normalize/Square:y:09triplet/dot_1/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(i
$triplet/dot_1/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
"triplet/dot_1/l2_normalize/MaximumMaximum'triplet/dot_1/l2_normalize/Sum:output:0-triplet/dot_1/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
 triplet/dot_1/l2_normalize/RsqrtRsqrt&triplet/dot_1/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
triplet/dot_1/l2_normalizeMul'triplet/backbone/dense/BiasAdd:output:0$triplet/dot_1/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@�
#triplet/dot_1/l2_normalize_1/SquareSquare)triplet/backbone/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:���������@t
2triplet/dot_1/l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
 triplet/dot_1/l2_normalize_1/SumSum'triplet/dot_1/l2_normalize_1/Square:y:0;triplet/dot_1/l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(k
&triplet/dot_1/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
$triplet/dot_1/l2_normalize_1/MaximumMaximum)triplet/dot_1/l2_normalize_1/Sum:output:0/triplet/dot_1/l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:����������
"triplet/dot_1/l2_normalize_1/RsqrtRsqrt(triplet/dot_1/l2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:����������
triplet/dot_1/l2_normalize_1Mul)triplet/backbone/dense/BiasAdd_1:output:0&triplet/dot_1/l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:���������@^
triplet/dot_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
triplet/dot_1/ExpandDims
ExpandDimstriplet/dot_1/l2_normalize:z:0%triplet/dot_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������@`
triplet/dot_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
triplet/dot_1/ExpandDims_1
ExpandDims triplet/dot_1/l2_normalize_1:z:0'triplet/dot_1/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:���������@�
triplet/dot_1/MatMulBatchMatMulV2!triplet/dot_1/ExpandDims:output:0#triplet/dot_1/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������`
triplet/dot_1/ShapeShapetriplet/dot_1/MatMul:output:0*
T0*
_output_shapes
:�
triplet/dot_1/SqueezeSqueezetriplet/dot_1/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
a
triplet/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
triplet/concatenate/concatConcatV2triplet/dot/Squeeze:output:0triplet/dot_1/Squeeze:output:0(triplet/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������r
IdentityIdentity#triplet/concatenate/concat:output:0^NoOp*
T0*'
_output_shapes
:����������%
NoOpNoOpE^triplet/backbone/batch_normalization/FusedBatchNormV3/ReadVariableOpG^triplet/backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_1G^triplet/backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOpI^triplet/backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1G^triplet/backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOpI^triplet/backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_14^triplet/backbone/batch_normalization/ReadVariableOp6^triplet/backbone/batch_normalization/ReadVariableOp_16^triplet/backbone/batch_normalization/ReadVariableOp_26^triplet/backbone/batch_normalization/ReadVariableOp_36^triplet/backbone/batch_normalization/ReadVariableOp_46^triplet/backbone/batch_normalization/ReadVariableOp_5G^triplet/backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOpI^triplet/backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1I^triplet/backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpK^triplet/backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1I^triplet/backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOpK^triplet/backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_16^triplet/backbone/batch_normalization_1/ReadVariableOp8^triplet/backbone/batch_normalization_1/ReadVariableOp_18^triplet/backbone/batch_normalization_1/ReadVariableOp_28^triplet/backbone/batch_normalization_1/ReadVariableOp_38^triplet/backbone/batch_normalization_1/ReadVariableOp_48^triplet/backbone/batch_normalization_1/ReadVariableOp_5G^triplet/backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOpI^triplet/backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1I^triplet/backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpK^triplet/backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1I^triplet/backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOpK^triplet/backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_16^triplet/backbone/batch_normalization_2/ReadVariableOp8^triplet/backbone/batch_normalization_2/ReadVariableOp_18^triplet/backbone/batch_normalization_2/ReadVariableOp_28^triplet/backbone/batch_normalization_2/ReadVariableOp_38^triplet/backbone/batch_normalization_2/ReadVariableOp_48^triplet/backbone/batch_normalization_2/ReadVariableOp_5G^triplet/backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOpI^triplet/backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1I^triplet/backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpK^triplet/backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1I^triplet/backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOpK^triplet/backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_16^triplet/backbone/batch_normalization_3/ReadVariableOp8^triplet/backbone/batch_normalization_3/ReadVariableOp_18^triplet/backbone/batch_normalization_3/ReadVariableOp_28^triplet/backbone/batch_normalization_3/ReadVariableOp_38^triplet/backbone/batch_normalization_3/ReadVariableOp_48^triplet/backbone/batch_normalization_3/ReadVariableOp_5/^triplet/backbone/conv2d/BiasAdd/ReadVariableOp1^triplet/backbone/conv2d/BiasAdd_1/ReadVariableOp1^triplet/backbone/conv2d/BiasAdd_2/ReadVariableOp.^triplet/backbone/conv2d/Conv2D/ReadVariableOp0^triplet/backbone/conv2d/Conv2D_1/ReadVariableOp0^triplet/backbone/conv2d/Conv2D_2/ReadVariableOp1^triplet/backbone/conv2d_1/BiasAdd/ReadVariableOp3^triplet/backbone/conv2d_1/BiasAdd_1/ReadVariableOp3^triplet/backbone/conv2d_1/BiasAdd_2/ReadVariableOp0^triplet/backbone/conv2d_1/Conv2D/ReadVariableOp2^triplet/backbone/conv2d_1/Conv2D_1/ReadVariableOp2^triplet/backbone/conv2d_1/Conv2D_2/ReadVariableOp1^triplet/backbone/conv2d_2/BiasAdd/ReadVariableOp3^triplet/backbone/conv2d_2/BiasAdd_1/ReadVariableOp3^triplet/backbone/conv2d_2/BiasAdd_2/ReadVariableOp0^triplet/backbone/conv2d_2/Conv2D/ReadVariableOp2^triplet/backbone/conv2d_2/Conv2D_1/ReadVariableOp2^triplet/backbone/conv2d_2/Conv2D_2/ReadVariableOp1^triplet/backbone/conv2d_3/BiasAdd/ReadVariableOp3^triplet/backbone/conv2d_3/BiasAdd_1/ReadVariableOp3^triplet/backbone/conv2d_3/BiasAdd_2/ReadVariableOp0^triplet/backbone/conv2d_3/Conv2D/ReadVariableOp2^triplet/backbone/conv2d_3/Conv2D_1/ReadVariableOp2^triplet/backbone/conv2d_3/Conv2D_2/ReadVariableOp.^triplet/backbone/dense/BiasAdd/ReadVariableOp0^triplet/backbone/dense/BiasAdd_1/ReadVariableOp0^triplet/backbone/dense/BiasAdd_2/ReadVariableOp-^triplet/backbone/dense/MatMul/ReadVariableOp/^triplet/backbone/dense/MatMul_1/ReadVariableOp/^triplet/backbone/dense/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������~:����������~:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 2�
Dtriplet/backbone/batch_normalization/FusedBatchNormV3/ReadVariableOpDtriplet/backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp2�
Ftriplet/backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ftriplet/backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_12�
Ftriplet/backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOpFtriplet/backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp2�
Htriplet/backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1Htriplet/backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_12�
Ftriplet/backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOpFtriplet/backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp2�
Htriplet/backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_1Htriplet/backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_12j
3triplet/backbone/batch_normalization/ReadVariableOp3triplet/backbone/batch_normalization/ReadVariableOp2n
5triplet/backbone/batch_normalization/ReadVariableOp_15triplet/backbone/batch_normalization/ReadVariableOp_12n
5triplet/backbone/batch_normalization/ReadVariableOp_25triplet/backbone/batch_normalization/ReadVariableOp_22n
5triplet/backbone/batch_normalization/ReadVariableOp_35triplet/backbone/batch_normalization/ReadVariableOp_32n
5triplet/backbone/batch_normalization/ReadVariableOp_45triplet/backbone/batch_normalization/ReadVariableOp_42n
5triplet/backbone/batch_normalization/ReadVariableOp_55triplet/backbone/batch_normalization/ReadVariableOp_52�
Ftriplet/backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOpFtriplet/backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2�
Htriplet/backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Htriplet/backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12�
Htriplet/backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpHtriplet/backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp2�
Jtriplet/backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1Jtriplet/backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_12�
Htriplet/backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOpHtriplet/backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp2�
Jtriplet/backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_1Jtriplet/backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_12n
5triplet/backbone/batch_normalization_1/ReadVariableOp5triplet/backbone/batch_normalization_1/ReadVariableOp2r
7triplet/backbone/batch_normalization_1/ReadVariableOp_17triplet/backbone/batch_normalization_1/ReadVariableOp_12r
7triplet/backbone/batch_normalization_1/ReadVariableOp_27triplet/backbone/batch_normalization_1/ReadVariableOp_22r
7triplet/backbone/batch_normalization_1/ReadVariableOp_37triplet/backbone/batch_normalization_1/ReadVariableOp_32r
7triplet/backbone/batch_normalization_1/ReadVariableOp_47triplet/backbone/batch_normalization_1/ReadVariableOp_42r
7triplet/backbone/batch_normalization_1/ReadVariableOp_57triplet/backbone/batch_normalization_1/ReadVariableOp_52�
Ftriplet/backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOpFtriplet/backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2�
Htriplet/backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Htriplet/backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12�
Htriplet/backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpHtriplet/backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp2�
Jtriplet/backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1Jtriplet/backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_12�
Htriplet/backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOpHtriplet/backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp2�
Jtriplet/backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_1Jtriplet/backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_12n
5triplet/backbone/batch_normalization_2/ReadVariableOp5triplet/backbone/batch_normalization_2/ReadVariableOp2r
7triplet/backbone/batch_normalization_2/ReadVariableOp_17triplet/backbone/batch_normalization_2/ReadVariableOp_12r
7triplet/backbone/batch_normalization_2/ReadVariableOp_27triplet/backbone/batch_normalization_2/ReadVariableOp_22r
7triplet/backbone/batch_normalization_2/ReadVariableOp_37triplet/backbone/batch_normalization_2/ReadVariableOp_32r
7triplet/backbone/batch_normalization_2/ReadVariableOp_47triplet/backbone/batch_normalization_2/ReadVariableOp_42r
7triplet/backbone/batch_normalization_2/ReadVariableOp_57triplet/backbone/batch_normalization_2/ReadVariableOp_52�
Ftriplet/backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOpFtriplet/backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
Htriplet/backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Htriplet/backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12�
Htriplet/backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpHtriplet/backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp2�
Jtriplet/backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1Jtriplet/backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_12�
Htriplet/backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOpHtriplet/backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp2�
Jtriplet/backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_1Jtriplet/backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_12n
5triplet/backbone/batch_normalization_3/ReadVariableOp5triplet/backbone/batch_normalization_3/ReadVariableOp2r
7triplet/backbone/batch_normalization_3/ReadVariableOp_17triplet/backbone/batch_normalization_3/ReadVariableOp_12r
7triplet/backbone/batch_normalization_3/ReadVariableOp_27triplet/backbone/batch_normalization_3/ReadVariableOp_22r
7triplet/backbone/batch_normalization_3/ReadVariableOp_37triplet/backbone/batch_normalization_3/ReadVariableOp_32r
7triplet/backbone/batch_normalization_3/ReadVariableOp_47triplet/backbone/batch_normalization_3/ReadVariableOp_42r
7triplet/backbone/batch_normalization_3/ReadVariableOp_57triplet/backbone/batch_normalization_3/ReadVariableOp_52`
.triplet/backbone/conv2d/BiasAdd/ReadVariableOp.triplet/backbone/conv2d/BiasAdd/ReadVariableOp2d
0triplet/backbone/conv2d/BiasAdd_1/ReadVariableOp0triplet/backbone/conv2d/BiasAdd_1/ReadVariableOp2d
0triplet/backbone/conv2d/BiasAdd_2/ReadVariableOp0triplet/backbone/conv2d/BiasAdd_2/ReadVariableOp2^
-triplet/backbone/conv2d/Conv2D/ReadVariableOp-triplet/backbone/conv2d/Conv2D/ReadVariableOp2b
/triplet/backbone/conv2d/Conv2D_1/ReadVariableOp/triplet/backbone/conv2d/Conv2D_1/ReadVariableOp2b
/triplet/backbone/conv2d/Conv2D_2/ReadVariableOp/triplet/backbone/conv2d/Conv2D_2/ReadVariableOp2d
0triplet/backbone/conv2d_1/BiasAdd/ReadVariableOp0triplet/backbone/conv2d_1/BiasAdd/ReadVariableOp2h
2triplet/backbone/conv2d_1/BiasAdd_1/ReadVariableOp2triplet/backbone/conv2d_1/BiasAdd_1/ReadVariableOp2h
2triplet/backbone/conv2d_1/BiasAdd_2/ReadVariableOp2triplet/backbone/conv2d_1/BiasAdd_2/ReadVariableOp2b
/triplet/backbone/conv2d_1/Conv2D/ReadVariableOp/triplet/backbone/conv2d_1/Conv2D/ReadVariableOp2f
1triplet/backbone/conv2d_1/Conv2D_1/ReadVariableOp1triplet/backbone/conv2d_1/Conv2D_1/ReadVariableOp2f
1triplet/backbone/conv2d_1/Conv2D_2/ReadVariableOp1triplet/backbone/conv2d_1/Conv2D_2/ReadVariableOp2d
0triplet/backbone/conv2d_2/BiasAdd/ReadVariableOp0triplet/backbone/conv2d_2/BiasAdd/ReadVariableOp2h
2triplet/backbone/conv2d_2/BiasAdd_1/ReadVariableOp2triplet/backbone/conv2d_2/BiasAdd_1/ReadVariableOp2h
2triplet/backbone/conv2d_2/BiasAdd_2/ReadVariableOp2triplet/backbone/conv2d_2/BiasAdd_2/ReadVariableOp2b
/triplet/backbone/conv2d_2/Conv2D/ReadVariableOp/triplet/backbone/conv2d_2/Conv2D/ReadVariableOp2f
1triplet/backbone/conv2d_2/Conv2D_1/ReadVariableOp1triplet/backbone/conv2d_2/Conv2D_1/ReadVariableOp2f
1triplet/backbone/conv2d_2/Conv2D_2/ReadVariableOp1triplet/backbone/conv2d_2/Conv2D_2/ReadVariableOp2d
0triplet/backbone/conv2d_3/BiasAdd/ReadVariableOp0triplet/backbone/conv2d_3/BiasAdd/ReadVariableOp2h
2triplet/backbone/conv2d_3/BiasAdd_1/ReadVariableOp2triplet/backbone/conv2d_3/BiasAdd_1/ReadVariableOp2h
2triplet/backbone/conv2d_3/BiasAdd_2/ReadVariableOp2triplet/backbone/conv2d_3/BiasAdd_2/ReadVariableOp2b
/triplet/backbone/conv2d_3/Conv2D/ReadVariableOp/triplet/backbone/conv2d_3/Conv2D/ReadVariableOp2f
1triplet/backbone/conv2d_3/Conv2D_1/ReadVariableOp1triplet/backbone/conv2d_3/Conv2D_1/ReadVariableOp2f
1triplet/backbone/conv2d_3/Conv2D_2/ReadVariableOp1triplet/backbone/conv2d_3/Conv2D_2/ReadVariableOp2^
-triplet/backbone/dense/BiasAdd/ReadVariableOp-triplet/backbone/dense/BiasAdd/ReadVariableOp2b
/triplet/backbone/dense/BiasAdd_1/ReadVariableOp/triplet/backbone/dense/BiasAdd_1/ReadVariableOp2b
/triplet/backbone/dense/BiasAdd_2/ReadVariableOp/triplet/backbone/dense/BiasAdd_2/ReadVariableOp2\
,triplet/backbone/dense/MatMul/ReadVariableOp,triplet/backbone/dense/MatMul/ReadVariableOp2`
.triplet/backbone/dense/MatMul_1/ReadVariableOp.triplet/backbone/dense/MatMul_1/ReadVariableOp2`
.triplet/backbone/dense/MatMul_2/ReadVariableOp.triplet/backbone/dense/MatMul_2/ReadVariableOp:^ Z
0
_output_shapes
:����������~
&
_user_specified_nameanchor_input:`\
0
_output_shapes
:����������~
(
_user_specified_namepositive_input:`\
0
_output_shapes
:����������~
(
_user_specified_namenegative_input
�
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1183999

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1187712

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1184577

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@?@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������@?@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@?@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������@?@
 
_user_specified_nameinputs
�
Q
%__inference_dot_layer_call_fn_1186929
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dot_layer_call_and_return_conditional_losses_1185203`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs/1
�
M
1__inference_max_pooling2d_2_layer_call_fn_1187509

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1184238h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� @:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
�
�
*__inference_backbone_layer_call_fn_1186721

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	�@

unknown_24:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_backbone_layer_call_and_return_conditional_losses_1184797o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������~
 
_user_specified_nameinputs
�

�
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1184194

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:��������� @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
�
c
G__inference_activation_layer_call_and_return_conditional_losses_1184120

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:����������~@c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������~@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������~@:X T
0
_output_shapes
:����������~@
 
_user_specified_nameinputs
�
�
*__inference_conv2d_2_layer_call_fn_1187355

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1184194w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
�
�
*__inference_backbone_layer_call_fn_1184382
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	�@

unknown_24:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_backbone_layer_call_and_return_conditional_losses_1184327o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:����������~
!
_user_specified_name	input_1
�	
�
B__inference_dense_layer_call_and_return_conditional_losses_1184320

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
t
H__inference_concatenate_layer_call_and_return_conditional_losses_1187000
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�

�
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1184250

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1187346

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:��������� @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@?@:W S
/
_output_shapes
:���������@?@
 
_user_specified_nameinputs
�

�
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1187192

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������@?@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@?@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@?@
 
_user_specified_nameinputs
�
J
.__inference_activation_1_layer_call_fn_1187321

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1184176h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@?@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@?@:W S
/
_output_shapes
:���������@?@
 
_user_specified_nameinputs
�.
�
D__inference_triplet_layer_call_and_return_conditional_losses_1185498

inputs
inputs_1
inputs_2*
backbone_1185387:@
backbone_1185389:@
backbone_1185391:@
backbone_1185393:@
backbone_1185395:@
backbone_1185397:@*
backbone_1185399:@@
backbone_1185401:@
backbone_1185403:@
backbone_1185405:@
backbone_1185407:@
backbone_1185409:@*
backbone_1185411:@@
backbone_1185413:@
backbone_1185415:@
backbone_1185417:@
backbone_1185419:@
backbone_1185421:@*
backbone_1185423:@@
backbone_1185425:@
backbone_1185427:@
backbone_1185429:@
backbone_1185431:@
backbone_1185433:@#
backbone_1185435:	�@
backbone_1185437:@
identity�� backbone/StatefulPartitionedCall�"backbone/StatefulPartitionedCall_1�"backbone/StatefulPartitionedCall_2�
 backbone/StatefulPartitionedCallStatefulPartitionedCallinputsbackbone_1185387backbone_1185389backbone_1185391backbone_1185393backbone_1185395backbone_1185397backbone_1185399backbone_1185401backbone_1185403backbone_1185405backbone_1185407backbone_1185409backbone_1185411backbone_1185413backbone_1185415backbone_1185417backbone_1185419backbone_1185421backbone_1185423backbone_1185425backbone_1185427backbone_1185429backbone_1185431backbone_1185433backbone_1185435backbone_1185437*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_backbone_layer_call_and_return_conditional_losses_1184797�
"backbone/StatefulPartitionedCall_1StatefulPartitionedCallinputs_2backbone_1185387backbone_1185389backbone_1185391backbone_1185393backbone_1185395backbone_1185397backbone_1185399backbone_1185401backbone_1185403backbone_1185405backbone_1185407backbone_1185409backbone_1185411backbone_1185413backbone_1185415backbone_1185417backbone_1185419backbone_1185421backbone_1185423backbone_1185425backbone_1185427backbone_1185429backbone_1185431backbone_1185433backbone_1185435backbone_1185437!^backbone/StatefulPartitionedCall*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_backbone_layer_call_and_return_conditional_losses_1184797�
"backbone/StatefulPartitionedCall_2StatefulPartitionedCallinputs_1backbone_1185387backbone_1185389backbone_1185391backbone_1185393backbone_1185395backbone_1185397backbone_1185399backbone_1185401backbone_1185403backbone_1185405backbone_1185407backbone_1185409backbone_1185411backbone_1185413backbone_1185415backbone_1185417backbone_1185419backbone_1185421backbone_1185423backbone_1185425backbone_1185427backbone_1185429backbone_1185431backbone_1185433backbone_1185435backbone_1185437#^backbone/StatefulPartitionedCall_1*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_backbone_layer_call_and_return_conditional_losses_1184797�
dot/PartitionedCallPartitionedCall)backbone/StatefulPartitionedCall:output:0+backbone/StatefulPartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dot_layer_call_and_return_conditional_losses_1185203�
dot_1/PartitionedCallPartitionedCall)backbone/StatefulPartitionedCall:output:0+backbone/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dot_1_layer_call_and_return_conditional_losses_1185231�
concatenate/PartitionedCallPartitionedCalldot/PartitionedCall:output:0dot_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1185240s
IdentityIdentity$concatenate/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^backbone/StatefulPartitionedCall#^backbone/StatefulPartitionedCall_1#^backbone/StatefulPartitionedCall_2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������~:����������~:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 backbone/StatefulPartitionedCall backbone/StatefulPartitionedCall2H
"backbone/StatefulPartitionedCall_1"backbone/StatefulPartitionedCall_12H
"backbone/StatefulPartitionedCall_2"backbone/StatefulPartitionedCall_2:X T
0
_output_shapes
:����������~
 
_user_specified_nameinputs:XT
0
_output_shapes
:����������~
 
_user_specified_nameinputs:XT
0
_output_shapes
:����������~
 
_user_specified_nameinputs
�
n
B__inference_dot_1_layer_call_and_return_conditional_losses_1186987
inputs_0
inputs_1
identityY
l2_normalize/SquareSquareinputs_0*
T0*'
_output_shapes
:���������@d
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������g
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������g
l2_normalizeMulinputs_0l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@[
l2_normalize_1/SquareSquareinputs_1*
T0*'
_output_shapes
:���������@f
$l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize_1/SumSuml2_normalize_1/Square:y:0-l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(]
l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize_1/MaximumMaximuml2_normalize_1/Sum:output:0!l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������k
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:���������k
l2_normalize_1Mulinputs_1l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:���������@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :y

ExpandDims
ExpandDimsl2_normalize:z:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������@R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsl2_normalize_1:z:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:���������@y
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:���������D
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:l
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
X
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs/1
�
c
G__inference_activation_layer_call_and_return_conditional_losses_1187153

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:����������~@c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������~@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������~@:X T
0
_output_shapes
:����������~@
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1184238

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� @:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
�
K
/__inference_max_pooling2d_layer_call_fn_1187158

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1183822�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_2_layer_call_fn_1187378

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1183923�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1183923

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
J
.__inference_activation_2_layer_call_fn_1187494

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_1184232h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� @:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_3_layer_call_fn_1187551

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1183999�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_2_layer_call_fn_1187404

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1184217w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1184642

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:����������~@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:����������~@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������~@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:����������~@
 
_user_specified_nameinputs
�
�
*__inference_backbone_layer_call_fn_1184909
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	�@

unknown_24:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_backbone_layer_call_and_return_conditional_losses_1184797o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:����������~
!
_user_specified_name	input_1
�
H
,__inference_activation_layer_call_fn_1187148

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������~@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1184120i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������~@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������~@:X T
0
_output_shapes
:����������~@
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1187143

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:����������~@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:����������~@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������~@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:����������~@
 
_user_specified_nameinputs
�
e
I__inference_activation_3_layer_call_and_return_conditional_losses_1187672

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1187489

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� @:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:��������� @�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1184300

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
l
B__inference_dot_1_layer_call_and_return_conditional_losses_1185231

inputs
inputs_1
identityW
l2_normalize/SquareSquareinputs*
T0*'
_output_shapes
:���������@d
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������g
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������e
l2_normalizeMulinputsl2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@[
l2_normalize_1/SquareSquareinputs_1*
T0*'
_output_shapes
:���������@f
$l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize_1/SumSuml2_normalize_1/Square:y:0-l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(]
l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize_1/MaximumMaximuml2_normalize_1/Sum:output:0!l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������k
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:���������k
l2_normalize_1Mulinputs_1l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:���������@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :y

ExpandDims
ExpandDimsl2_normalize:z:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������@R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsl2_normalize_1:z:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:���������@y
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:���������D
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:l
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
X
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_3_layer_call_fn_1187682

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1184294h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1183802

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_1187723

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�U
�
E__inference_backbone_layer_call_and_return_conditional_losses_1184797

inputs(
conv2d_1184725:@
conv2d_1184727:@)
batch_normalization_1184730:@)
batch_normalization_1184732:@)
batch_normalization_1184734:@)
batch_normalization_1184736:@*
conv2d_1_1184741:@@
conv2d_1_1184743:@+
batch_normalization_1_1184746:@+
batch_normalization_1_1184748:@+
batch_normalization_1_1184750:@+
batch_normalization_1_1184752:@*
conv2d_2_1184757:@@
conv2d_2_1184759:@+
batch_normalization_2_1184762:@+
batch_normalization_2_1184764:@+
batch_normalization_2_1184766:@+
batch_normalization_2_1184768:@*
conv2d_3_1184773:@@
conv2d_3_1184775:@+
batch_normalization_3_1184778:@+
batch_normalization_3_1184780:@+
batch_normalization_3_1184782:@+
batch_normalization_3_1184784:@ 
dense_1184791:	�@
dense_1184793:@
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�dense/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1184725conv2d_1184727*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������~@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1184082�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1184730batch_normalization_1184732batch_normalization_1184734batch_normalization_1184736*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������~@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1184642�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������~@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1184120�
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1184126�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_1184741conv2d_1_1184743*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1184138�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_1184746batch_normalization_1_1184748batch_normalization_1_1184750batch_normalization_1_1184752*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1184577�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1184176�
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1184182�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_1184757conv2d_2_1184759*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1184194�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_1184762batch_normalization_2_1184764batch_normalization_2_1184766batch_normalization_2_1184768*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1184512�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_1184232�
max_pooling2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1184238�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_1184773conv2d_3_1184775*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1184250�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_1184778batch_normalization_3_1184780batch_normalization_3_1184782batch_normalization_3_1184784*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1184447�
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_1184288�
max_pooling2d_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1184294�
max_pooling2d_4/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1184300�
flatten/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1184308�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1184791dense_1184793*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1184320u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:X T
0
_output_shapes
:����������~
 
_user_specified_nameinputs
�

�
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1187538

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
C__inference_conv2d_layer_call_and_return_conditional_losses_1187019

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:����������~@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������~: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������~
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1187453

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1187471

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� @:@:@:@:@:*
epsilon%o�:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:��������� @�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1187608

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_3_layer_call_fn_1187577

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1184273w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�U
�
E__inference_backbone_layer_call_and_return_conditional_losses_1185059
input_1(
conv2d_1184987:@
conv2d_1184989:@)
batch_normalization_1184992:@)
batch_normalization_1184994:@)
batch_normalization_1184996:@)
batch_normalization_1184998:@*
conv2d_1_1185003:@@
conv2d_1_1185005:@+
batch_normalization_1_1185008:@+
batch_normalization_1_1185010:@+
batch_normalization_1_1185012:@+
batch_normalization_1_1185014:@*
conv2d_2_1185019:@@
conv2d_2_1185021:@+
batch_normalization_2_1185024:@+
batch_normalization_2_1185026:@+
batch_normalization_2_1185028:@+
batch_normalization_2_1185030:@*
conv2d_3_1185035:@@
conv2d_3_1185037:@+
batch_normalization_3_1185040:@+
batch_normalization_3_1185042:@+
batch_normalization_3_1185044:@+
batch_normalization_3_1185046:@ 
dense_1185053:	�@
dense_1185055:@
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�dense/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_1184987conv2d_1184989*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������~@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1184082�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1184992batch_normalization_1184994batch_normalization_1184996batch_normalization_1184998*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������~@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1184642�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������~@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1184120�
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1184126�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_1185003conv2d_1_1185005*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1184138�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_1185008batch_normalization_1_1185010batch_normalization_1_1185012batch_normalization_1_1185014*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1184577�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1184176�
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1184182�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_1185019conv2d_2_1185021*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1184194�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_1185024batch_normalization_2_1185026batch_normalization_2_1185028batch_normalization_2_1185030*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1184512�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_1184232�
max_pooling2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1184238�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_1185035conv2d_3_1185037*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1184250�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_1185040batch_normalization_3_1185042batch_normalization_3_1185044batch_normalization_3_1185046*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1184447�
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_1184288�
max_pooling2d_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1184294�
max_pooling2d_4/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1184300�
flatten/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1184308�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1185053dense_1185055*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1184320u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Y U
0
_output_shapes
:����������~
!
_user_specified_name	input_1
��
�
 __inference__traced_save_1187974
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*�
value�B�FB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*�
value�B�FB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *T
dtypesJ
H2F	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : : : :@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:	�@:@: : :@:@:@:@:@@:@:@:@:@@:@:@:@:@@:@:@:@:	�@:@:@:@:@:@:@@:@:@:@:@@:@:@:@:@@:@:@:@:	�@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?
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
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	�@: 

_output_shapes
:@: 

_output_shapes
: :!

_output_shapes
: :,"(
&
_output_shapes
:@: #

_output_shapes
:@: $

_output_shapes
:@: %

_output_shapes
:@:,&(
&
_output_shapes
:@@: '

_output_shapes
:@: (

_output_shapes
:@: )

_output_shapes
:@:,*(
&
_output_shapes
:@@: +

_output_shapes
:@: ,

_output_shapes
:@: -

_output_shapes
:@:,.(
&
_output_shapes
:@@: /

_output_shapes
:@: 0

_output_shapes
:@: 1

_output_shapes
:@:%2!

_output_shapes
:	�@: 3

_output_shapes
:@:,4(
&
_output_shapes
:@: 5

_output_shapes
:@: 6

_output_shapes
:@: 7

_output_shapes
:@:,8(
&
_output_shapes
:@@: 9

_output_shapes
:@: :

_output_shapes
:@: ;

_output_shapes
:@:,<(
&
_output_shapes
:@@: =

_output_shapes
:@: >

_output_shapes
:@: ?

_output_shapes
:@:,@(
&
_output_shapes
:@@: A

_output_shapes
:@: B

_output_shapes
:@: C

_output_shapes
:@:%D!

_output_shapes
:	�@: E

_output_shapes
:@:F

_output_shapes
: 
�
h
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1183898

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
(__inference_conv2d_layer_call_fn_1187009

inputs!
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������~@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1184082x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������~@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������~: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������~
 
_user_specified_nameinputs
�z
�
E__inference_backbone_layer_call_and_return_conditional_losses_1186822

inputs?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@;
-batch_normalization_3_readvariableop_resource:@=
/batch_normalization_3_readvariableop_1_resource:@L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:@7
$dense_matmul_readvariableop_resource:	�@3
%dense_biasadd_readvariableop_resource:@
identity��3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@*
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:����������~@:@:@:@:@:*
epsilon%o�:*
is_training( |
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������~@�
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:���������@?@*
ksize
*
paddingVALID*
strides
�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@*
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@?@:@:@:@:@:*
epsilon%o�:*
is_training( 
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@?@�
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� @:@:@:@:@:*
epsilon%o�:*
is_training( 
activation_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� @�
max_pooling2d_2/MaxPoolMaxPoolactivation_2/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 
activation_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@�
max_pooling2d_3/MaxPoolMaxPoolactivation_3/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
max_pooling2d_4/MaxPoolMaxPool max_pooling2d_3/MaxPool:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshape max_pooling2d_4/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:�����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@�	
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:X T
0
_output_shapes
:����������~
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_layer_call_fn_1187071

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������~@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1184642x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������~@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������~@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������~@
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1187707

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
e
I__inference_activation_1_layer_call_and_return_conditional_losses_1187326

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������@?@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@?@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@?@:W S
/
_output_shapes
:���������@?@
 
_user_specified_nameinputs
�U
�
E__inference_backbone_layer_call_and_return_conditional_losses_1184984
input_1(
conv2d_1184912:@
conv2d_1184914:@)
batch_normalization_1184917:@)
batch_normalization_1184919:@)
batch_normalization_1184921:@)
batch_normalization_1184923:@*
conv2d_1_1184928:@@
conv2d_1_1184930:@+
batch_normalization_1_1184933:@+
batch_normalization_1_1184935:@+
batch_normalization_1_1184937:@+
batch_normalization_1_1184939:@*
conv2d_2_1184944:@@
conv2d_2_1184946:@+
batch_normalization_2_1184949:@+
batch_normalization_2_1184951:@+
batch_normalization_2_1184953:@+
batch_normalization_2_1184955:@*
conv2d_3_1184960:@@
conv2d_3_1184962:@+
batch_normalization_3_1184965:@+
batch_normalization_3_1184967:@+
batch_normalization_3_1184969:@+
batch_normalization_3_1184971:@ 
dense_1184978:	�@
dense_1184980:@
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�dense/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_1184912conv2d_1184914*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������~@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1184082�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1184917batch_normalization_1184919batch_normalization_1184921batch_normalization_1184923*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������~@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1184105�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������~@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1184120�
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1184126�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_1184928conv2d_1_1184930*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1184138�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_1184933batch_normalization_1_1184935batch_normalization_1_1184937batch_normalization_1_1184939*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1184161�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1184176�
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1184182�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_1184944conv2d_2_1184946*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1184194�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_1184949batch_normalization_2_1184951batch_normalization_2_1184953batch_normalization_2_1184955*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1184217�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_1184232�
max_pooling2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1184238�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_1184960conv2d_3_1184962*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1184250�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_1184965batch_normalization_3_1184967batch_normalization_3_1184969batch_normalization_3_1184971*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1184273�
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_1184288�
max_pooling2d_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1184294�
max_pooling2d_4/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1184300�
flatten/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1184308�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1184978dense_1184980*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1184320u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Y U
0
_output_shapes
:����������~
!
_user_specified_name	input_1
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1184217

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� @:@:@:@:@:*
epsilon%o�:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:��������� @�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
�	
�
B__inference_dense_layer_call_and_return_conditional_losses_1187742

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1184050

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1184138

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������@?@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@?@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@?@
 
_user_specified_nameinputs
۞
�9
D__inference_triplet_layer_call_and_return_conditional_losses_1186607
inputs_0
inputs_1
inputs_2H
.backbone_conv2d_conv2d_readvariableop_resource:@=
/backbone_conv2d_biasadd_readvariableop_resource:@B
4backbone_batch_normalization_readvariableop_resource:@D
6backbone_batch_normalization_readvariableop_1_resource:@S
Ebackbone_batch_normalization_fusedbatchnormv3_readvariableop_resource:@U
Gbackbone_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@J
0backbone_conv2d_1_conv2d_readvariableop_resource:@@?
1backbone_conv2d_1_biasadd_readvariableop_resource:@D
6backbone_batch_normalization_1_readvariableop_resource:@F
8backbone_batch_normalization_1_readvariableop_1_resource:@U
Gbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@W
Ibackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@J
0backbone_conv2d_2_conv2d_readvariableop_resource:@@?
1backbone_conv2d_2_biasadd_readvariableop_resource:@D
6backbone_batch_normalization_2_readvariableop_resource:@F
8backbone_batch_normalization_2_readvariableop_1_resource:@U
Gbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@W
Ibackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@J
0backbone_conv2d_3_conv2d_readvariableop_resource:@@?
1backbone_conv2d_3_biasadd_readvariableop_resource:@D
6backbone_batch_normalization_3_readvariableop_resource:@F
8backbone_batch_normalization_3_readvariableop_1_resource:@U
Gbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:@W
Ibackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:@@
-backbone_dense_matmul_readvariableop_resource:	�@<
.backbone_dense_biasadd_readvariableop_resource:@
identity��+backbone/batch_normalization/AssignNewValue�-backbone/batch_normalization/AssignNewValue_1�-backbone/batch_normalization/AssignNewValue_2�-backbone/batch_normalization/AssignNewValue_3�-backbone/batch_normalization/AssignNewValue_4�-backbone/batch_normalization/AssignNewValue_5�<backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp�>backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�>backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp�@backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1�>backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp�@backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_1�+backbone/batch_normalization/ReadVariableOp�-backbone/batch_normalization/ReadVariableOp_1�-backbone/batch_normalization/ReadVariableOp_2�-backbone/batch_normalization/ReadVariableOp_3�-backbone/batch_normalization/ReadVariableOp_4�-backbone/batch_normalization/ReadVariableOp_5�-backbone/batch_normalization_1/AssignNewValue�/backbone/batch_normalization_1/AssignNewValue_1�/backbone/batch_normalization_1/AssignNewValue_2�/backbone/batch_normalization_1/AssignNewValue_3�/backbone/batch_normalization_1/AssignNewValue_4�/backbone/batch_normalization_1/AssignNewValue_5�>backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�@backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�@backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp�Bbackbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1�@backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp�Bbackbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_1�-backbone/batch_normalization_1/ReadVariableOp�/backbone/batch_normalization_1/ReadVariableOp_1�/backbone/batch_normalization_1/ReadVariableOp_2�/backbone/batch_normalization_1/ReadVariableOp_3�/backbone/batch_normalization_1/ReadVariableOp_4�/backbone/batch_normalization_1/ReadVariableOp_5�-backbone/batch_normalization_2/AssignNewValue�/backbone/batch_normalization_2/AssignNewValue_1�/backbone/batch_normalization_2/AssignNewValue_2�/backbone/batch_normalization_2/AssignNewValue_3�/backbone/batch_normalization_2/AssignNewValue_4�/backbone/batch_normalization_2/AssignNewValue_5�>backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�@backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�@backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp�Bbackbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1�@backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp�Bbackbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_1�-backbone/batch_normalization_2/ReadVariableOp�/backbone/batch_normalization_2/ReadVariableOp_1�/backbone/batch_normalization_2/ReadVariableOp_2�/backbone/batch_normalization_2/ReadVariableOp_3�/backbone/batch_normalization_2/ReadVariableOp_4�/backbone/batch_normalization_2/ReadVariableOp_5�-backbone/batch_normalization_3/AssignNewValue�/backbone/batch_normalization_3/AssignNewValue_1�/backbone/batch_normalization_3/AssignNewValue_2�/backbone/batch_normalization_3/AssignNewValue_3�/backbone/batch_normalization_3/AssignNewValue_4�/backbone/batch_normalization_3/AssignNewValue_5�>backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�@backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�@backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp�Bbackbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1�@backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp�Bbackbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_1�-backbone/batch_normalization_3/ReadVariableOp�/backbone/batch_normalization_3/ReadVariableOp_1�/backbone/batch_normalization_3/ReadVariableOp_2�/backbone/batch_normalization_3/ReadVariableOp_3�/backbone/batch_normalization_3/ReadVariableOp_4�/backbone/batch_normalization_3/ReadVariableOp_5�&backbone/conv2d/BiasAdd/ReadVariableOp�(backbone/conv2d/BiasAdd_1/ReadVariableOp�(backbone/conv2d/BiasAdd_2/ReadVariableOp�%backbone/conv2d/Conv2D/ReadVariableOp�'backbone/conv2d/Conv2D_1/ReadVariableOp�'backbone/conv2d/Conv2D_2/ReadVariableOp�(backbone/conv2d_1/BiasAdd/ReadVariableOp�*backbone/conv2d_1/BiasAdd_1/ReadVariableOp�*backbone/conv2d_1/BiasAdd_2/ReadVariableOp�'backbone/conv2d_1/Conv2D/ReadVariableOp�)backbone/conv2d_1/Conv2D_1/ReadVariableOp�)backbone/conv2d_1/Conv2D_2/ReadVariableOp�(backbone/conv2d_2/BiasAdd/ReadVariableOp�*backbone/conv2d_2/BiasAdd_1/ReadVariableOp�*backbone/conv2d_2/BiasAdd_2/ReadVariableOp�'backbone/conv2d_2/Conv2D/ReadVariableOp�)backbone/conv2d_2/Conv2D_1/ReadVariableOp�)backbone/conv2d_2/Conv2D_2/ReadVariableOp�(backbone/conv2d_3/BiasAdd/ReadVariableOp�*backbone/conv2d_3/BiasAdd_1/ReadVariableOp�*backbone/conv2d_3/BiasAdd_2/ReadVariableOp�'backbone/conv2d_3/Conv2D/ReadVariableOp�)backbone/conv2d_3/Conv2D_1/ReadVariableOp�)backbone/conv2d_3/Conv2D_2/ReadVariableOp�%backbone/dense/BiasAdd/ReadVariableOp�'backbone/dense/BiasAdd_1/ReadVariableOp�'backbone/dense/BiasAdd_2/ReadVariableOp�$backbone/dense/MatMul/ReadVariableOp�&backbone/dense/MatMul_1/ReadVariableOp�&backbone/dense/MatMul_2/ReadVariableOp�
%backbone/conv2d/Conv2D/ReadVariableOpReadVariableOp.backbone_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
backbone/conv2d/Conv2DConv2Dinputs_0-backbone/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@*
paddingSAME*
strides
�
&backbone/conv2d/BiasAdd/ReadVariableOpReadVariableOp/backbone_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d/BiasAddBiasAddbackbone/conv2d/Conv2D:output:0.backbone/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@�
+backbone/batch_normalization/ReadVariableOpReadVariableOp4backbone_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
-backbone/batch_normalization/ReadVariableOp_1ReadVariableOp6backbone_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
<backbone/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpEbackbone_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
>backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGbackbone_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
-backbone/batch_normalization/FusedBatchNormV3FusedBatchNormV3 backbone/conv2d/BiasAdd:output:03backbone/batch_normalization/ReadVariableOp:value:05backbone/batch_normalization/ReadVariableOp_1:value:0Dbackbone/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Fbackbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:����������~@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
+backbone/batch_normalization/AssignNewValueAssignVariableOpEbackbone_batch_normalization_fusedbatchnormv3_readvariableop_resource:backbone/batch_normalization/FusedBatchNormV3:batch_mean:0=^backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
-backbone/batch_normalization/AssignNewValue_1AssignVariableOpGbackbone_batch_normalization_fusedbatchnormv3_readvariableop_1_resource>backbone/batch_normalization/FusedBatchNormV3:batch_variance:0?^backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0�
backbone/activation/ReluRelu1backbone/batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������~@�
backbone/max_pooling2d/MaxPoolMaxPool&backbone/activation/Relu:activations:0*/
_output_shapes
:���������@?@*
ksize
*
paddingVALID*
strides
�
'backbone/conv2d_1/Conv2D/ReadVariableOpReadVariableOp0backbone_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_1/Conv2DConv2D'backbone/max_pooling2d/MaxPool:output:0/backbone/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@*
paddingSAME*
strides
�
(backbone/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp1backbone_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_1/BiasAddBiasAdd!backbone/conv2d_1/Conv2D:output:00backbone/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@�
-backbone/batch_normalization_1/ReadVariableOpReadVariableOp6backbone_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_1/ReadVariableOp_1ReadVariableOp8backbone_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
>backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpGbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3"backbone/conv2d_1/BiasAdd:output:05backbone/batch_normalization_1/ReadVariableOp:value:07backbone/batch_normalization_1/ReadVariableOp_1:value:0Fbackbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Hbackbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@?@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
-backbone/batch_normalization_1/AssignNewValueAssignVariableOpGbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_resource<backbone/batch_normalization_1/FusedBatchNormV3:batch_mean:0?^backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
/backbone/batch_normalization_1/AssignNewValue_1AssignVariableOpIbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource@backbone/batch_normalization_1/FusedBatchNormV3:batch_variance:0A^backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0�
backbone/activation_1/ReluRelu3backbone/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@?@�
 backbone/max_pooling2d_1/MaxPoolMaxPool(backbone/activation_1/Relu:activations:0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
�
'backbone/conv2d_2/Conv2D/ReadVariableOpReadVariableOp0backbone_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_2/Conv2DConv2D)backbone/max_pooling2d_1/MaxPool:output:0/backbone/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
�
(backbone/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp1backbone_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_2/BiasAddBiasAdd!backbone/conv2d_2/Conv2D:output:00backbone/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @�
-backbone/batch_normalization_2/ReadVariableOpReadVariableOp6backbone_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_2/ReadVariableOp_1ReadVariableOp8backbone_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
>backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpGbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3"backbone/conv2d_2/BiasAdd:output:05backbone/batch_normalization_2/ReadVariableOp:value:07backbone/batch_normalization_2/ReadVariableOp_1:value:0Fbackbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Hbackbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� @:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
-backbone/batch_normalization_2/AssignNewValueAssignVariableOpGbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_resource<backbone/batch_normalization_2/FusedBatchNormV3:batch_mean:0?^backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
/backbone/batch_normalization_2/AssignNewValue_1AssignVariableOpIbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource@backbone/batch_normalization_2/FusedBatchNormV3:batch_variance:0A^backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0�
backbone/activation_2/ReluRelu3backbone/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� @�
 backbone/max_pooling2d_2/MaxPoolMaxPool(backbone/activation_2/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
'backbone/conv2d_3/Conv2D/ReadVariableOpReadVariableOp0backbone_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_3/Conv2DConv2D)backbone/max_pooling2d_2/MaxPool:output:0/backbone/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
(backbone/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp1backbone_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_3/BiasAddBiasAdd!backbone/conv2d_3/Conv2D:output:00backbone/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
-backbone/batch_normalization_3/ReadVariableOpReadVariableOp6backbone_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_3/ReadVariableOp_1ReadVariableOp8backbone_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
>backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpGbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3"backbone/conv2d_3/BiasAdd:output:05backbone/batch_normalization_3/ReadVariableOp:value:07backbone/batch_normalization_3/ReadVariableOp_1:value:0Fbackbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Hbackbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
-backbone/batch_normalization_3/AssignNewValueAssignVariableOpGbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_resource<backbone/batch_normalization_3/FusedBatchNormV3:batch_mean:0?^backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
/backbone/batch_normalization_3/AssignNewValue_1AssignVariableOpIbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource@backbone/batch_normalization_3/FusedBatchNormV3:batch_variance:0A^backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0�
backbone/activation_3/ReluRelu3backbone/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@�
 backbone/max_pooling2d_3/MaxPoolMaxPool(backbone/activation_3/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
 backbone/max_pooling2d_4/MaxPoolMaxPool)backbone/max_pooling2d_3/MaxPool:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
g
backbone/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
backbone/flatten/ReshapeReshape)backbone/max_pooling2d_4/MaxPool:output:0backbone/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
$backbone/dense/MatMul/ReadVariableOpReadVariableOp-backbone_dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
backbone/dense/MatMulMatMul!backbone/flatten/Reshape:output:0,backbone/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
%backbone/dense/BiasAdd/ReadVariableOpReadVariableOp.backbone_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/dense/BiasAddBiasAddbackbone/dense/MatMul:product:0-backbone/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'backbone/conv2d/Conv2D_1/ReadVariableOpReadVariableOp.backbone_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
backbone/conv2d/Conv2D_1Conv2Dinputs_2/backbone/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@*
paddingSAME*
strides
�
(backbone/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp/backbone_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d/BiasAdd_1BiasAdd!backbone/conv2d/Conv2D_1:output:00backbone/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@�
-backbone/batch_normalization/ReadVariableOp_2ReadVariableOp4backbone_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
-backbone/batch_normalization/ReadVariableOp_3ReadVariableOp6backbone_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
>backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEbackbone_batch_normalization_fusedbatchnormv3_readvariableop_resource,^backbone/batch_normalization/AssignNewValue*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGbackbone_batch_normalization_fusedbatchnormv3_readvariableop_1_resource.^backbone/batch_normalization/AssignNewValue_1*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization/FusedBatchNormV3_1FusedBatchNormV3"backbone/conv2d/BiasAdd_1:output:05backbone/batch_normalization/ReadVariableOp_2:value:05backbone/batch_normalization/ReadVariableOp_3:value:0Fbackbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp:value:0Hbackbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:����������~@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
-backbone/batch_normalization/AssignNewValue_2AssignVariableOpEbackbone_batch_normalization_fusedbatchnormv3_readvariableop_resource<backbone/batch_normalization/FusedBatchNormV3_1:batch_mean:0,^backbone/batch_normalization/AssignNewValue?^backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0�
-backbone/batch_normalization/AssignNewValue_3AssignVariableOpGbackbone_batch_normalization_fusedbatchnormv3_readvariableop_1_resource@backbone/batch_normalization/FusedBatchNormV3_1:batch_variance:0.^backbone/batch_normalization/AssignNewValue_1A^backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0�
backbone/activation/Relu_1Relu3backbone/batch_normalization/FusedBatchNormV3_1:y:0*
T0*0
_output_shapes
:����������~@�
 backbone/max_pooling2d/MaxPool_1MaxPool(backbone/activation/Relu_1:activations:0*/
_output_shapes
:���������@?@*
ksize
*
paddingVALID*
strides
�
)backbone/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp0backbone_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_1/Conv2D_1Conv2D)backbone/max_pooling2d/MaxPool_1:output:01backbone/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@*
paddingSAME*
strides
�
*backbone/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp1backbone_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_1/BiasAdd_1BiasAdd#backbone/conv2d_1/Conv2D_1:output:02backbone/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@�
/backbone/batch_normalization_1/ReadVariableOp_2ReadVariableOp6backbone_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_1/ReadVariableOp_3ReadVariableOp8backbone_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpReadVariableOpGbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_resource.^backbone/batch_normalization_1/AssignNewValue*
_output_shapes
:@*
dtype0�
Bbackbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource0^backbone/batch_normalization_1/AssignNewValue_1*
_output_shapes
:@*
dtype0�
1backbone/batch_normalization_1/FusedBatchNormV3_1FusedBatchNormV3$backbone/conv2d_1/BiasAdd_1:output:07backbone/batch_normalization_1/ReadVariableOp_2:value:07backbone/batch_normalization_1/ReadVariableOp_3:value:0Hbackbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp:value:0Jbackbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@?@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
/backbone/batch_normalization_1/AssignNewValue_2AssignVariableOpGbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_resource>backbone/batch_normalization_1/FusedBatchNormV3_1:batch_mean:0.^backbone/batch_normalization_1/AssignNewValueA^backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0�
/backbone/batch_normalization_1/AssignNewValue_3AssignVariableOpIbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceBbackbone/batch_normalization_1/FusedBatchNormV3_1:batch_variance:00^backbone/batch_normalization_1/AssignNewValue_1C^backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0�
backbone/activation_1/Relu_1Relu5backbone/batch_normalization_1/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:���������@?@�
"backbone/max_pooling2d_1/MaxPool_1MaxPool*backbone/activation_1/Relu_1:activations:0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
�
)backbone/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOp0backbone_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_2/Conv2D_1Conv2D+backbone/max_pooling2d_1/MaxPool_1:output:01backbone/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
�
*backbone/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOp1backbone_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_2/BiasAdd_1BiasAdd#backbone/conv2d_2/Conv2D_1:output:02backbone/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @�
/backbone/batch_normalization_2/ReadVariableOp_2ReadVariableOp6backbone_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_2/ReadVariableOp_3ReadVariableOp8backbone_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpReadVariableOpGbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_resource.^backbone/batch_normalization_2/AssignNewValue*
_output_shapes
:@*
dtype0�
Bbackbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource0^backbone/batch_normalization_2/AssignNewValue_1*
_output_shapes
:@*
dtype0�
1backbone/batch_normalization_2/FusedBatchNormV3_1FusedBatchNormV3$backbone/conv2d_2/BiasAdd_1:output:07backbone/batch_normalization_2/ReadVariableOp_2:value:07backbone/batch_normalization_2/ReadVariableOp_3:value:0Hbackbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp:value:0Jbackbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� @:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
/backbone/batch_normalization_2/AssignNewValue_2AssignVariableOpGbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_resource>backbone/batch_normalization_2/FusedBatchNormV3_1:batch_mean:0.^backbone/batch_normalization_2/AssignNewValueA^backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0�
/backbone/batch_normalization_2/AssignNewValue_3AssignVariableOpIbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceBbackbone/batch_normalization_2/FusedBatchNormV3_1:batch_variance:00^backbone/batch_normalization_2/AssignNewValue_1C^backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0�
backbone/activation_2/Relu_1Relu5backbone/batch_normalization_2/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:��������� @�
"backbone/max_pooling2d_2/MaxPool_1MaxPool*backbone/activation_2/Relu_1:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
)backbone/conv2d_3/Conv2D_1/ReadVariableOpReadVariableOp0backbone_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_3/Conv2D_1Conv2D+backbone/max_pooling2d_2/MaxPool_1:output:01backbone/conv2d_3/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
*backbone/conv2d_3/BiasAdd_1/ReadVariableOpReadVariableOp1backbone_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_3/BiasAdd_1BiasAdd#backbone/conv2d_3/Conv2D_1:output:02backbone/conv2d_3/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
/backbone/batch_normalization_3/ReadVariableOp_2ReadVariableOp6backbone_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_3/ReadVariableOp_3ReadVariableOp8backbone_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpReadVariableOpGbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_resource.^backbone/batch_normalization_3/AssignNewValue*
_output_shapes
:@*
dtype0�
Bbackbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource0^backbone/batch_normalization_3/AssignNewValue_1*
_output_shapes
:@*
dtype0�
1backbone/batch_normalization_3/FusedBatchNormV3_1FusedBatchNormV3$backbone/conv2d_3/BiasAdd_1:output:07backbone/batch_normalization_3/ReadVariableOp_2:value:07backbone/batch_normalization_3/ReadVariableOp_3:value:0Hbackbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp:value:0Jbackbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
/backbone/batch_normalization_3/AssignNewValue_2AssignVariableOpGbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_resource>backbone/batch_normalization_3/FusedBatchNormV3_1:batch_mean:0.^backbone/batch_normalization_3/AssignNewValueA^backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0�
/backbone/batch_normalization_3/AssignNewValue_3AssignVariableOpIbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceBbackbone/batch_normalization_3/FusedBatchNormV3_1:batch_variance:00^backbone/batch_normalization_3/AssignNewValue_1C^backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0�
backbone/activation_3/Relu_1Relu5backbone/batch_normalization_3/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:���������@�
"backbone/max_pooling2d_3/MaxPool_1MaxPool*backbone/activation_3/Relu_1:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
"backbone/max_pooling2d_4/MaxPool_1MaxPool+backbone/max_pooling2d_3/MaxPool_1:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
i
backbone/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"����   �
backbone/flatten/Reshape_1Reshape+backbone/max_pooling2d_4/MaxPool_1:output:0!backbone/flatten/Const_1:output:0*
T0*(
_output_shapes
:�����������
&backbone/dense/MatMul_1/ReadVariableOpReadVariableOp-backbone_dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
backbone/dense/MatMul_1MatMul#backbone/flatten/Reshape_1:output:0.backbone/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'backbone/dense/BiasAdd_1/ReadVariableOpReadVariableOp.backbone_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/dense/BiasAdd_1BiasAdd!backbone/dense/MatMul_1:product:0/backbone/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'backbone/conv2d/Conv2D_2/ReadVariableOpReadVariableOp.backbone_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
backbone/conv2d/Conv2D_2Conv2Dinputs_1/backbone/conv2d/Conv2D_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@*
paddingSAME*
strides
�
(backbone/conv2d/BiasAdd_2/ReadVariableOpReadVariableOp/backbone_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d/BiasAdd_2BiasAdd!backbone/conv2d/Conv2D_2:output:00backbone/conv2d/BiasAdd_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@�
-backbone/batch_normalization/ReadVariableOp_4ReadVariableOp4backbone_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
-backbone/batch_normalization/ReadVariableOp_5ReadVariableOp6backbone_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
>backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOpReadVariableOpEbackbone_batch_normalization_fusedbatchnormv3_readvariableop_resource.^backbone/batch_normalization/AssignNewValue_2*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_1ReadVariableOpGbackbone_batch_normalization_fusedbatchnormv3_readvariableop_1_resource.^backbone/batch_normalization/AssignNewValue_3*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization/FusedBatchNormV3_2FusedBatchNormV3"backbone/conv2d/BiasAdd_2:output:05backbone/batch_normalization/ReadVariableOp_4:value:05backbone/batch_normalization/ReadVariableOp_5:value:0Fbackbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp:value:0Hbackbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:����������~@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
-backbone/batch_normalization/AssignNewValue_4AssignVariableOpEbackbone_batch_normalization_fusedbatchnormv3_readvariableop_resource<backbone/batch_normalization/FusedBatchNormV3_2:batch_mean:0.^backbone/batch_normalization/AssignNewValue_2?^backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp*
_output_shapes
 *
dtype0�
-backbone/batch_normalization/AssignNewValue_5AssignVariableOpGbackbone_batch_normalization_fusedbatchnormv3_readvariableop_1_resource@backbone/batch_normalization/FusedBatchNormV3_2:batch_variance:0.^backbone/batch_normalization/AssignNewValue_3A^backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_1*
_output_shapes
 *
dtype0�
backbone/activation/Relu_2Relu3backbone/batch_normalization/FusedBatchNormV3_2:y:0*
T0*0
_output_shapes
:����������~@�
 backbone/max_pooling2d/MaxPool_2MaxPool(backbone/activation/Relu_2:activations:0*/
_output_shapes
:���������@?@*
ksize
*
paddingVALID*
strides
�
)backbone/conv2d_1/Conv2D_2/ReadVariableOpReadVariableOp0backbone_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_1/Conv2D_2Conv2D)backbone/max_pooling2d/MaxPool_2:output:01backbone/conv2d_1/Conv2D_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@*
paddingSAME*
strides
�
*backbone/conv2d_1/BiasAdd_2/ReadVariableOpReadVariableOp1backbone_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_1/BiasAdd_2BiasAdd#backbone/conv2d_1/Conv2D_2:output:02backbone/conv2d_1/BiasAdd_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@�
/backbone/batch_normalization_1/ReadVariableOp_4ReadVariableOp6backbone_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_1/ReadVariableOp_5ReadVariableOp8backbone_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOpReadVariableOpGbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_resource0^backbone/batch_normalization_1/AssignNewValue_2*
_output_shapes
:@*
dtype0�
Bbackbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource0^backbone/batch_normalization_1/AssignNewValue_3*
_output_shapes
:@*
dtype0�
1backbone/batch_normalization_1/FusedBatchNormV3_2FusedBatchNormV3$backbone/conv2d_1/BiasAdd_2:output:07backbone/batch_normalization_1/ReadVariableOp_4:value:07backbone/batch_normalization_1/ReadVariableOp_5:value:0Hbackbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp:value:0Jbackbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@?@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
/backbone/batch_normalization_1/AssignNewValue_4AssignVariableOpGbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_resource>backbone/batch_normalization_1/FusedBatchNormV3_2:batch_mean:00^backbone/batch_normalization_1/AssignNewValue_2A^backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp*
_output_shapes
 *
dtype0�
/backbone/batch_normalization_1/AssignNewValue_5AssignVariableOpIbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceBbackbone/batch_normalization_1/FusedBatchNormV3_2:batch_variance:00^backbone/batch_normalization_1/AssignNewValue_3C^backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_1*
_output_shapes
 *
dtype0�
backbone/activation_1/Relu_2Relu5backbone/batch_normalization_1/FusedBatchNormV3_2:y:0*
T0*/
_output_shapes
:���������@?@�
"backbone/max_pooling2d_1/MaxPool_2MaxPool*backbone/activation_1/Relu_2:activations:0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
�
)backbone/conv2d_2/Conv2D_2/ReadVariableOpReadVariableOp0backbone_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_2/Conv2D_2Conv2D+backbone/max_pooling2d_1/MaxPool_2:output:01backbone/conv2d_2/Conv2D_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
�
*backbone/conv2d_2/BiasAdd_2/ReadVariableOpReadVariableOp1backbone_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_2/BiasAdd_2BiasAdd#backbone/conv2d_2/Conv2D_2:output:02backbone/conv2d_2/BiasAdd_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @�
/backbone/batch_normalization_2/ReadVariableOp_4ReadVariableOp6backbone_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_2/ReadVariableOp_5ReadVariableOp8backbone_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOpReadVariableOpGbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_resource0^backbone/batch_normalization_2/AssignNewValue_2*
_output_shapes
:@*
dtype0�
Bbackbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource0^backbone/batch_normalization_2/AssignNewValue_3*
_output_shapes
:@*
dtype0�
1backbone/batch_normalization_2/FusedBatchNormV3_2FusedBatchNormV3$backbone/conv2d_2/BiasAdd_2:output:07backbone/batch_normalization_2/ReadVariableOp_4:value:07backbone/batch_normalization_2/ReadVariableOp_5:value:0Hbackbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp:value:0Jbackbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� @:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
/backbone/batch_normalization_2/AssignNewValue_4AssignVariableOpGbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_resource>backbone/batch_normalization_2/FusedBatchNormV3_2:batch_mean:00^backbone/batch_normalization_2/AssignNewValue_2A^backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp*
_output_shapes
 *
dtype0�
/backbone/batch_normalization_2/AssignNewValue_5AssignVariableOpIbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceBbackbone/batch_normalization_2/FusedBatchNormV3_2:batch_variance:00^backbone/batch_normalization_2/AssignNewValue_3C^backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_1*
_output_shapes
 *
dtype0�
backbone/activation_2/Relu_2Relu5backbone/batch_normalization_2/FusedBatchNormV3_2:y:0*
T0*/
_output_shapes
:��������� @�
"backbone/max_pooling2d_2/MaxPool_2MaxPool*backbone/activation_2/Relu_2:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
)backbone/conv2d_3/Conv2D_2/ReadVariableOpReadVariableOp0backbone_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_3/Conv2D_2Conv2D+backbone/max_pooling2d_2/MaxPool_2:output:01backbone/conv2d_3/Conv2D_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
*backbone/conv2d_3/BiasAdd_2/ReadVariableOpReadVariableOp1backbone_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_3/BiasAdd_2BiasAdd#backbone/conv2d_3/Conv2D_2:output:02backbone/conv2d_3/BiasAdd_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
/backbone/batch_normalization_3/ReadVariableOp_4ReadVariableOp6backbone_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_3/ReadVariableOp_5ReadVariableOp8backbone_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOpReadVariableOpGbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_resource0^backbone/batch_normalization_3/AssignNewValue_2*
_output_shapes
:@*
dtype0�
Bbackbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource0^backbone/batch_normalization_3/AssignNewValue_3*
_output_shapes
:@*
dtype0�
1backbone/batch_normalization_3/FusedBatchNormV3_2FusedBatchNormV3$backbone/conv2d_3/BiasAdd_2:output:07backbone/batch_normalization_3/ReadVariableOp_4:value:07backbone/batch_normalization_3/ReadVariableOp_5:value:0Hbackbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp:value:0Jbackbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
/backbone/batch_normalization_3/AssignNewValue_4AssignVariableOpGbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_resource>backbone/batch_normalization_3/FusedBatchNormV3_2:batch_mean:00^backbone/batch_normalization_3/AssignNewValue_2A^backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp*
_output_shapes
 *
dtype0�
/backbone/batch_normalization_3/AssignNewValue_5AssignVariableOpIbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceBbackbone/batch_normalization_3/FusedBatchNormV3_2:batch_variance:00^backbone/batch_normalization_3/AssignNewValue_3C^backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_1*
_output_shapes
 *
dtype0�
backbone/activation_3/Relu_2Relu5backbone/batch_normalization_3/FusedBatchNormV3_2:y:0*
T0*/
_output_shapes
:���������@�
"backbone/max_pooling2d_3/MaxPool_2MaxPool*backbone/activation_3/Relu_2:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
"backbone/max_pooling2d_4/MaxPool_2MaxPool+backbone/max_pooling2d_3/MaxPool_2:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
i
backbone/flatten/Const_2Const*
_output_shapes
:*
dtype0*
valueB"����   �
backbone/flatten/Reshape_2Reshape+backbone/max_pooling2d_4/MaxPool_2:output:0!backbone/flatten/Const_2:output:0*
T0*(
_output_shapes
:�����������
&backbone/dense/MatMul_2/ReadVariableOpReadVariableOp-backbone_dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
backbone/dense/MatMul_2MatMul#backbone/flatten/Reshape_2:output:0.backbone/dense/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'backbone/dense/BiasAdd_2/ReadVariableOpReadVariableOp.backbone_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/dense/BiasAdd_2BiasAdd!backbone/dense/MatMul_2:product:0/backbone/dense/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@t
dot/l2_normalize/SquareSquarebackbone/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@h
&dot/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
dot/l2_normalize/SumSumdot/l2_normalize/Square:y:0/dot/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(_
dot/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
dot/l2_normalize/MaximumMaximumdot/l2_normalize/Sum:output:0#dot/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������o
dot/l2_normalize/RsqrtRsqrtdot/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
dot/l2_normalizeMulbackbone/dense/BiasAdd:output:0dot/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@x
dot/l2_normalize_1/SquareSquare!backbone/dense/BiasAdd_2:output:0*
T0*'
_output_shapes
:���������@j
(dot/l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
dot/l2_normalize_1/SumSumdot/l2_normalize_1/Square:y:01dot/l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(a
dot/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
dot/l2_normalize_1/MaximumMaximumdot/l2_normalize_1/Sum:output:0%dot/l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������s
dot/l2_normalize_1/RsqrtRsqrtdot/l2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:����������
dot/l2_normalize_1Mul!backbone/dense/BiasAdd_2:output:0dot/l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:���������@T
dot/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot/ExpandDims
ExpandDimsdot/l2_normalize:z:0dot/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������@V
dot/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot/ExpandDims_1
ExpandDimsdot/l2_normalize_1:z:0dot/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:���������@�

dot/MatMulBatchMatMulV2dot/ExpandDims:output:0dot/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������L
	dot/ShapeShapedot/MatMul:output:0*
T0*
_output_shapes
:t
dot/SqueezeSqueezedot/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
v
dot_1/l2_normalize/SquareSquarebackbone/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@j
(dot_1/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
dot_1/l2_normalize/SumSumdot_1/l2_normalize/Square:y:01dot_1/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(a
dot_1/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
dot_1/l2_normalize/MaximumMaximumdot_1/l2_normalize/Sum:output:0%dot_1/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������s
dot_1/l2_normalize/RsqrtRsqrtdot_1/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
dot_1/l2_normalizeMulbackbone/dense/BiasAdd:output:0dot_1/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@z
dot_1/l2_normalize_1/SquareSquare!backbone/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:���������@l
*dot_1/l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
dot_1/l2_normalize_1/SumSumdot_1/l2_normalize_1/Square:y:03dot_1/l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(c
dot_1/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
dot_1/l2_normalize_1/MaximumMaximum!dot_1/l2_normalize_1/Sum:output:0'dot_1/l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������w
dot_1/l2_normalize_1/RsqrtRsqrt dot_1/l2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:����������
dot_1/l2_normalize_1Mul!backbone/dense/BiasAdd_1:output:0dot_1/l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:���������@V
dot_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot_1/ExpandDims
ExpandDimsdot_1/l2_normalize:z:0dot_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������@X
dot_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot_1/ExpandDims_1
ExpandDimsdot_1/l2_normalize_1:z:0dot_1/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:���������@�
dot_1/MatMulBatchMatMulV2dot_1/ExpandDims:output:0dot_1/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������P
dot_1/ShapeShapedot_1/MatMul:output:0*
T0*
_output_shapes
:x
dot_1/SqueezeSqueezedot_1/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2dot/Squeeze:output:0dot_1/Squeeze:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������j
IdentityIdentityconcatenate/concat:output:0^NoOp*
T0*'
_output_shapes
:����������)
NoOpNoOp,^backbone/batch_normalization/AssignNewValue.^backbone/batch_normalization/AssignNewValue_1.^backbone/batch_normalization/AssignNewValue_2.^backbone/batch_normalization/AssignNewValue_3.^backbone/batch_normalization/AssignNewValue_4.^backbone/batch_normalization/AssignNewValue_5=^backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp?^backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?^backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOpA^backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1?^backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOpA^backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_1,^backbone/batch_normalization/ReadVariableOp.^backbone/batch_normalization/ReadVariableOp_1.^backbone/batch_normalization/ReadVariableOp_2.^backbone/batch_normalization/ReadVariableOp_3.^backbone/batch_normalization/ReadVariableOp_4.^backbone/batch_normalization/ReadVariableOp_5.^backbone/batch_normalization_1/AssignNewValue0^backbone/batch_normalization_1/AssignNewValue_10^backbone/batch_normalization_1/AssignNewValue_20^backbone/batch_normalization_1/AssignNewValue_30^backbone/batch_normalization_1/AssignNewValue_40^backbone/batch_normalization_1/AssignNewValue_5?^backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOpA^backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1A^backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpC^backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1A^backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOpC^backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_1.^backbone/batch_normalization_1/ReadVariableOp0^backbone/batch_normalization_1/ReadVariableOp_10^backbone/batch_normalization_1/ReadVariableOp_20^backbone/batch_normalization_1/ReadVariableOp_30^backbone/batch_normalization_1/ReadVariableOp_40^backbone/batch_normalization_1/ReadVariableOp_5.^backbone/batch_normalization_2/AssignNewValue0^backbone/batch_normalization_2/AssignNewValue_10^backbone/batch_normalization_2/AssignNewValue_20^backbone/batch_normalization_2/AssignNewValue_30^backbone/batch_normalization_2/AssignNewValue_40^backbone/batch_normalization_2/AssignNewValue_5?^backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOpA^backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1A^backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpC^backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1A^backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOpC^backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_1.^backbone/batch_normalization_2/ReadVariableOp0^backbone/batch_normalization_2/ReadVariableOp_10^backbone/batch_normalization_2/ReadVariableOp_20^backbone/batch_normalization_2/ReadVariableOp_30^backbone/batch_normalization_2/ReadVariableOp_40^backbone/batch_normalization_2/ReadVariableOp_5.^backbone/batch_normalization_3/AssignNewValue0^backbone/batch_normalization_3/AssignNewValue_10^backbone/batch_normalization_3/AssignNewValue_20^backbone/batch_normalization_3/AssignNewValue_30^backbone/batch_normalization_3/AssignNewValue_40^backbone/batch_normalization_3/AssignNewValue_5?^backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOpA^backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1A^backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpC^backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1A^backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOpC^backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_1.^backbone/batch_normalization_3/ReadVariableOp0^backbone/batch_normalization_3/ReadVariableOp_10^backbone/batch_normalization_3/ReadVariableOp_20^backbone/batch_normalization_3/ReadVariableOp_30^backbone/batch_normalization_3/ReadVariableOp_40^backbone/batch_normalization_3/ReadVariableOp_5'^backbone/conv2d/BiasAdd/ReadVariableOp)^backbone/conv2d/BiasAdd_1/ReadVariableOp)^backbone/conv2d/BiasAdd_2/ReadVariableOp&^backbone/conv2d/Conv2D/ReadVariableOp(^backbone/conv2d/Conv2D_1/ReadVariableOp(^backbone/conv2d/Conv2D_2/ReadVariableOp)^backbone/conv2d_1/BiasAdd/ReadVariableOp+^backbone/conv2d_1/BiasAdd_1/ReadVariableOp+^backbone/conv2d_1/BiasAdd_2/ReadVariableOp(^backbone/conv2d_1/Conv2D/ReadVariableOp*^backbone/conv2d_1/Conv2D_1/ReadVariableOp*^backbone/conv2d_1/Conv2D_2/ReadVariableOp)^backbone/conv2d_2/BiasAdd/ReadVariableOp+^backbone/conv2d_2/BiasAdd_1/ReadVariableOp+^backbone/conv2d_2/BiasAdd_2/ReadVariableOp(^backbone/conv2d_2/Conv2D/ReadVariableOp*^backbone/conv2d_2/Conv2D_1/ReadVariableOp*^backbone/conv2d_2/Conv2D_2/ReadVariableOp)^backbone/conv2d_3/BiasAdd/ReadVariableOp+^backbone/conv2d_3/BiasAdd_1/ReadVariableOp+^backbone/conv2d_3/BiasAdd_2/ReadVariableOp(^backbone/conv2d_3/Conv2D/ReadVariableOp*^backbone/conv2d_3/Conv2D_1/ReadVariableOp*^backbone/conv2d_3/Conv2D_2/ReadVariableOp&^backbone/dense/BiasAdd/ReadVariableOp(^backbone/dense/BiasAdd_1/ReadVariableOp(^backbone/dense/BiasAdd_2/ReadVariableOp%^backbone/dense/MatMul/ReadVariableOp'^backbone/dense/MatMul_1/ReadVariableOp'^backbone/dense/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������~:����������~:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+backbone/batch_normalization/AssignNewValue+backbone/batch_normalization/AssignNewValue2^
-backbone/batch_normalization/AssignNewValue_1-backbone/batch_normalization/AssignNewValue_12^
-backbone/batch_normalization/AssignNewValue_2-backbone/batch_normalization/AssignNewValue_22^
-backbone/batch_normalization/AssignNewValue_3-backbone/batch_normalization/AssignNewValue_32^
-backbone/batch_normalization/AssignNewValue_4-backbone/batch_normalization/AssignNewValue_42^
-backbone/batch_normalization/AssignNewValue_5-backbone/batch_normalization/AssignNewValue_52|
<backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp<backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp2�
>backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_1>backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_12�
>backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp>backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp2�
@backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1@backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_12�
>backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp>backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp2�
@backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_1@backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_12Z
+backbone/batch_normalization/ReadVariableOp+backbone/batch_normalization/ReadVariableOp2^
-backbone/batch_normalization/ReadVariableOp_1-backbone/batch_normalization/ReadVariableOp_12^
-backbone/batch_normalization/ReadVariableOp_2-backbone/batch_normalization/ReadVariableOp_22^
-backbone/batch_normalization/ReadVariableOp_3-backbone/batch_normalization/ReadVariableOp_32^
-backbone/batch_normalization/ReadVariableOp_4-backbone/batch_normalization/ReadVariableOp_42^
-backbone/batch_normalization/ReadVariableOp_5-backbone/batch_normalization/ReadVariableOp_52^
-backbone/batch_normalization_1/AssignNewValue-backbone/batch_normalization_1/AssignNewValue2b
/backbone/batch_normalization_1/AssignNewValue_1/backbone/batch_normalization_1/AssignNewValue_12b
/backbone/batch_normalization_1/AssignNewValue_2/backbone/batch_normalization_1/AssignNewValue_22b
/backbone/batch_normalization_1/AssignNewValue_3/backbone/batch_normalization_1/AssignNewValue_32b
/backbone/batch_normalization_1/AssignNewValue_4/backbone/batch_normalization_1/AssignNewValue_42b
/backbone/batch_normalization_1/AssignNewValue_5/backbone/batch_normalization_1/AssignNewValue_52�
>backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2�
@backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1@backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12�
@backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp@backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp2�
Bbackbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1Bbackbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_12�
@backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp@backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp2�
Bbackbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_1Bbackbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_12^
-backbone/batch_normalization_1/ReadVariableOp-backbone/batch_normalization_1/ReadVariableOp2b
/backbone/batch_normalization_1/ReadVariableOp_1/backbone/batch_normalization_1/ReadVariableOp_12b
/backbone/batch_normalization_1/ReadVariableOp_2/backbone/batch_normalization_1/ReadVariableOp_22b
/backbone/batch_normalization_1/ReadVariableOp_3/backbone/batch_normalization_1/ReadVariableOp_32b
/backbone/batch_normalization_1/ReadVariableOp_4/backbone/batch_normalization_1/ReadVariableOp_42b
/backbone/batch_normalization_1/ReadVariableOp_5/backbone/batch_normalization_1/ReadVariableOp_52^
-backbone/batch_normalization_2/AssignNewValue-backbone/batch_normalization_2/AssignNewValue2b
/backbone/batch_normalization_2/AssignNewValue_1/backbone/batch_normalization_2/AssignNewValue_12b
/backbone/batch_normalization_2/AssignNewValue_2/backbone/batch_normalization_2/AssignNewValue_22b
/backbone/batch_normalization_2/AssignNewValue_3/backbone/batch_normalization_2/AssignNewValue_32b
/backbone/batch_normalization_2/AssignNewValue_4/backbone/batch_normalization_2/AssignNewValue_42b
/backbone/batch_normalization_2/AssignNewValue_5/backbone/batch_normalization_2/AssignNewValue_52�
>backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp>backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2�
@backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1@backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12�
@backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp@backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp2�
Bbackbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1Bbackbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_12�
@backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp@backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp2�
Bbackbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_1Bbackbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_12^
-backbone/batch_normalization_2/ReadVariableOp-backbone/batch_normalization_2/ReadVariableOp2b
/backbone/batch_normalization_2/ReadVariableOp_1/backbone/batch_normalization_2/ReadVariableOp_12b
/backbone/batch_normalization_2/ReadVariableOp_2/backbone/batch_normalization_2/ReadVariableOp_22b
/backbone/batch_normalization_2/ReadVariableOp_3/backbone/batch_normalization_2/ReadVariableOp_32b
/backbone/batch_normalization_2/ReadVariableOp_4/backbone/batch_normalization_2/ReadVariableOp_42b
/backbone/batch_normalization_2/ReadVariableOp_5/backbone/batch_normalization_2/ReadVariableOp_52^
-backbone/batch_normalization_3/AssignNewValue-backbone/batch_normalization_3/AssignNewValue2b
/backbone/batch_normalization_3/AssignNewValue_1/backbone/batch_normalization_3/AssignNewValue_12b
/backbone/batch_normalization_3/AssignNewValue_2/backbone/batch_normalization_3/AssignNewValue_22b
/backbone/batch_normalization_3/AssignNewValue_3/backbone/batch_normalization_3/AssignNewValue_32b
/backbone/batch_normalization_3/AssignNewValue_4/backbone/batch_normalization_3/AssignNewValue_42b
/backbone/batch_normalization_3/AssignNewValue_5/backbone/batch_normalization_3/AssignNewValue_52�
>backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp>backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
@backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1@backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12�
@backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp@backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp2�
Bbackbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1Bbackbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_12�
@backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp@backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp2�
Bbackbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_1Bbackbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_12^
-backbone/batch_normalization_3/ReadVariableOp-backbone/batch_normalization_3/ReadVariableOp2b
/backbone/batch_normalization_3/ReadVariableOp_1/backbone/batch_normalization_3/ReadVariableOp_12b
/backbone/batch_normalization_3/ReadVariableOp_2/backbone/batch_normalization_3/ReadVariableOp_22b
/backbone/batch_normalization_3/ReadVariableOp_3/backbone/batch_normalization_3/ReadVariableOp_32b
/backbone/batch_normalization_3/ReadVariableOp_4/backbone/batch_normalization_3/ReadVariableOp_42b
/backbone/batch_normalization_3/ReadVariableOp_5/backbone/batch_normalization_3/ReadVariableOp_52P
&backbone/conv2d/BiasAdd/ReadVariableOp&backbone/conv2d/BiasAdd/ReadVariableOp2T
(backbone/conv2d/BiasAdd_1/ReadVariableOp(backbone/conv2d/BiasAdd_1/ReadVariableOp2T
(backbone/conv2d/BiasAdd_2/ReadVariableOp(backbone/conv2d/BiasAdd_2/ReadVariableOp2N
%backbone/conv2d/Conv2D/ReadVariableOp%backbone/conv2d/Conv2D/ReadVariableOp2R
'backbone/conv2d/Conv2D_1/ReadVariableOp'backbone/conv2d/Conv2D_1/ReadVariableOp2R
'backbone/conv2d/Conv2D_2/ReadVariableOp'backbone/conv2d/Conv2D_2/ReadVariableOp2T
(backbone/conv2d_1/BiasAdd/ReadVariableOp(backbone/conv2d_1/BiasAdd/ReadVariableOp2X
*backbone/conv2d_1/BiasAdd_1/ReadVariableOp*backbone/conv2d_1/BiasAdd_1/ReadVariableOp2X
*backbone/conv2d_1/BiasAdd_2/ReadVariableOp*backbone/conv2d_1/BiasAdd_2/ReadVariableOp2R
'backbone/conv2d_1/Conv2D/ReadVariableOp'backbone/conv2d_1/Conv2D/ReadVariableOp2V
)backbone/conv2d_1/Conv2D_1/ReadVariableOp)backbone/conv2d_1/Conv2D_1/ReadVariableOp2V
)backbone/conv2d_1/Conv2D_2/ReadVariableOp)backbone/conv2d_1/Conv2D_2/ReadVariableOp2T
(backbone/conv2d_2/BiasAdd/ReadVariableOp(backbone/conv2d_2/BiasAdd/ReadVariableOp2X
*backbone/conv2d_2/BiasAdd_1/ReadVariableOp*backbone/conv2d_2/BiasAdd_1/ReadVariableOp2X
*backbone/conv2d_2/BiasAdd_2/ReadVariableOp*backbone/conv2d_2/BiasAdd_2/ReadVariableOp2R
'backbone/conv2d_2/Conv2D/ReadVariableOp'backbone/conv2d_2/Conv2D/ReadVariableOp2V
)backbone/conv2d_2/Conv2D_1/ReadVariableOp)backbone/conv2d_2/Conv2D_1/ReadVariableOp2V
)backbone/conv2d_2/Conv2D_2/ReadVariableOp)backbone/conv2d_2/Conv2D_2/ReadVariableOp2T
(backbone/conv2d_3/BiasAdd/ReadVariableOp(backbone/conv2d_3/BiasAdd/ReadVariableOp2X
*backbone/conv2d_3/BiasAdd_1/ReadVariableOp*backbone/conv2d_3/BiasAdd_1/ReadVariableOp2X
*backbone/conv2d_3/BiasAdd_2/ReadVariableOp*backbone/conv2d_3/BiasAdd_2/ReadVariableOp2R
'backbone/conv2d_3/Conv2D/ReadVariableOp'backbone/conv2d_3/Conv2D/ReadVariableOp2V
)backbone/conv2d_3/Conv2D_1/ReadVariableOp)backbone/conv2d_3/Conv2D_1/ReadVariableOp2V
)backbone/conv2d_3/Conv2D_2/ReadVariableOp)backbone/conv2d_3/Conv2D_2/ReadVariableOp2N
%backbone/dense/BiasAdd/ReadVariableOp%backbone/dense/BiasAdd/ReadVariableOp2R
'backbone/dense/BiasAdd_1/ReadVariableOp'backbone/dense/BiasAdd_1/ReadVariableOp2R
'backbone/dense/BiasAdd_2/ReadVariableOp'backbone/dense/BiasAdd_2/ReadVariableOp2L
$backbone/dense/MatMul/ReadVariableOp$backbone/dense/MatMul/ReadVariableOp2P
&backbone/dense/MatMul_1/ReadVariableOp&backbone/dense/MatMul_1/ReadVariableOp2P
&backbone/dense/MatMul_2/ReadVariableOp&backbone/dense/MatMul_2/ReadVariableOp:Z V
0
_output_shapes
:����������~
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:����������~
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:����������~
"
_user_specified_name
inputs/2
�
�
*__inference_backbone_layer_call_fn_1186664

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	�@

unknown_24:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_backbone_layer_call_and_return_conditional_losses_1184327o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������~
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1184294

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
*__inference_conv2d_1_layer_call_fn_1187182

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1184138w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@?@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@?@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@?@
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1187341

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1184182

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:��������� @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@?@:W S
/
_output_shapes
:���������@?@
 
_user_specified_nameinputs
Ε
�
E__inference_backbone_layer_call_and_return_conditional_losses_1186923

inputs?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@;
-batch_normalization_3_readvariableop_resource:@=
/batch_normalization_3_readvariableop_1_resource:@L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:@7
$dense_matmul_readvariableop_resource:	�@3
%dense_biasadd_readvariableop_resource:@
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�$batch_normalization_1/AssignNewValue�&batch_normalization_1/AssignNewValue_1�5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�$batch_normalization_2/AssignNewValue�&batch_normalization_2/AssignNewValue_1�5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�$batch_normalization_3/AssignNewValue�&batch_normalization_3/AssignNewValue_1�5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@*
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:����������~@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0|
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������~@�
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:���������@?@*
ksize
*
paddingVALID*
strides
�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@*
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@?@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@?@�
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� @:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� @�
max_pooling2d_2/MaxPoolMaxPoolactivation_2/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@�
max_pooling2d_3/MaxPoolMaxPoolactivation_3/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
max_pooling2d_4/MaxPoolMaxPool max_pooling2d_3/MaxPool:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshape max_pooling2d_4/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:�����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:X T
0
_output_shapes
:����������~
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1184030

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1183878

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1184062

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
Y
-__inference_concatenate_layer_call_fn_1186993
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1185240`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
e
I__inference_activation_1_layer_call_and_return_conditional_losses_1184176

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������@?@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@?@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@?@:W S
/
_output_shapes
:���������@?@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1184161

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@?@:@:@:@:@:*
epsilon%o�:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������@?@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@?@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������@?@
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_1_layer_call_fn_1187205

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1183847�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
J
.__inference_activation_3_layer_call_fn_1187667

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_1184288h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1187262

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1187644

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
)__inference_triplet_layer_call_fn_1185612
anchor_input
positive_input
negative_input!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	�@

unknown_24:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallanchor_inputpositive_inputnegative_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_triplet_layer_call_and_return_conditional_losses_1185498o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������~:����������~:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:����������~
&
_user_specified_nameanchor_input:`\
0
_output_shapes
:����������~
(
_user_specified_namepositive_input:`\
0
_output_shapes
:����������~
(
_user_specified_namenegative_input
�	
�
5__inference_batch_normalization_layer_call_fn_1187045

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1183802�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_1_layer_call_fn_1187231

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1184161w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@?@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@?@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@?@
 
_user_specified_nameinputs
��
�-
#__inference__traced_restore_1188191
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: :
 assignvariableop_5_conv2d_kernel:@,
assignvariableop_6_conv2d_bias:@:
,assignvariableop_7_batch_normalization_gamma:@9
+assignvariableop_8_batch_normalization_beta:@@
2assignvariableop_9_batch_normalization_moving_mean:@E
7assignvariableop_10_batch_normalization_moving_variance:@=
#assignvariableop_11_conv2d_1_kernel:@@/
!assignvariableop_12_conv2d_1_bias:@=
/assignvariableop_13_batch_normalization_1_gamma:@<
.assignvariableop_14_batch_normalization_1_beta:@C
5assignvariableop_15_batch_normalization_1_moving_mean:@G
9assignvariableop_16_batch_normalization_1_moving_variance:@=
#assignvariableop_17_conv2d_2_kernel:@@/
!assignvariableop_18_conv2d_2_bias:@=
/assignvariableop_19_batch_normalization_2_gamma:@<
.assignvariableop_20_batch_normalization_2_beta:@C
5assignvariableop_21_batch_normalization_2_moving_mean:@G
9assignvariableop_22_batch_normalization_2_moving_variance:@=
#assignvariableop_23_conv2d_3_kernel:@@/
!assignvariableop_24_conv2d_3_bias:@=
/assignvariableop_25_batch_normalization_3_gamma:@<
.assignvariableop_26_batch_normalization_3_beta:@C
5assignvariableop_27_batch_normalization_3_moving_mean:@G
9assignvariableop_28_batch_normalization_3_moving_variance:@3
 assignvariableop_29_dense_kernel:	�@,
assignvariableop_30_dense_bias:@#
assignvariableop_31_total: #
assignvariableop_32_count: B
(assignvariableop_33_adam_conv2d_kernel_m:@4
&assignvariableop_34_adam_conv2d_bias_m:@B
4assignvariableop_35_adam_batch_normalization_gamma_m:@A
3assignvariableop_36_adam_batch_normalization_beta_m:@D
*assignvariableop_37_adam_conv2d_1_kernel_m:@@6
(assignvariableop_38_adam_conv2d_1_bias_m:@D
6assignvariableop_39_adam_batch_normalization_1_gamma_m:@C
5assignvariableop_40_adam_batch_normalization_1_beta_m:@D
*assignvariableop_41_adam_conv2d_2_kernel_m:@@6
(assignvariableop_42_adam_conv2d_2_bias_m:@D
6assignvariableop_43_adam_batch_normalization_2_gamma_m:@C
5assignvariableop_44_adam_batch_normalization_2_beta_m:@D
*assignvariableop_45_adam_conv2d_3_kernel_m:@@6
(assignvariableop_46_adam_conv2d_3_bias_m:@D
6assignvariableop_47_adam_batch_normalization_3_gamma_m:@C
5assignvariableop_48_adam_batch_normalization_3_beta_m:@:
'assignvariableop_49_adam_dense_kernel_m:	�@3
%assignvariableop_50_adam_dense_bias_m:@B
(assignvariableop_51_adam_conv2d_kernel_v:@4
&assignvariableop_52_adam_conv2d_bias_v:@B
4assignvariableop_53_adam_batch_normalization_gamma_v:@A
3assignvariableop_54_adam_batch_normalization_beta_v:@D
*assignvariableop_55_adam_conv2d_1_kernel_v:@@6
(assignvariableop_56_adam_conv2d_1_bias_v:@D
6assignvariableop_57_adam_batch_normalization_1_gamma_v:@C
5assignvariableop_58_adam_batch_normalization_1_beta_v:@D
*assignvariableop_59_adam_conv2d_2_kernel_v:@@6
(assignvariableop_60_adam_conv2d_2_bias_v:@D
6assignvariableop_61_adam_batch_normalization_2_gamma_v:@C
5assignvariableop_62_adam_batch_normalization_2_beta_v:@D
*assignvariableop_63_adam_conv2d_3_kernel_v:@@6
(assignvariableop_64_adam_conv2d_3_bias_v:@D
6assignvariableop_65_adam_batch_normalization_3_gamma_v:@C
5assignvariableop_66_adam_batch_normalization_3_beta_v:@:
'assignvariableop_67_adam_dense_kernel_v:	�@3
%assignvariableop_68_adam_dense_bias_v:@
identity_70��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*�
value�B�FB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*�
value�B�FB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*T
dtypesJ
H2F	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv2d_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp,assignvariableop_7_batch_normalization_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp+assignvariableop_8_batch_normalization_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp2assignvariableop_9_batch_normalization_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp7assignvariableop_10_batch_normalization_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv2d_1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_1_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp.assignvariableop_14_batch_normalization_1_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp5assignvariableop_15_batch_normalization_1_moving_meanIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp9assignvariableop_16_batch_normalization_1_moving_varianceIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp#assignvariableop_17_conv2d_2_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp!assignvariableop_18_conv2d_2_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp/assignvariableop_19_batch_normalization_2_gammaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp.assignvariableop_20_batch_normalization_2_betaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp5assignvariableop_21_batch_normalization_2_moving_meanIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp9assignvariableop_22_batch_normalization_2_moving_varianceIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp#assignvariableop_23_conv2d_3_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp!assignvariableop_24_conv2d_3_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp/assignvariableop_25_batch_normalization_3_gammaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp.assignvariableop_26_batch_normalization_3_betaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp5assignvariableop_27_batch_normalization_3_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp9assignvariableop_28_batch_normalization_3_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp assignvariableop_29_dense_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_dense_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpassignvariableop_31_totalIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_countIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_conv2d_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_conv2d_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_batch_normalization_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp3assignvariableop_36_adam_batch_normalization_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_batch_normalization_1_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp5assignvariableop_40_adam_batch_normalization_1_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_2_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_2_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_batch_normalization_2_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp5assignvariableop_44_adam_batch_normalization_2_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv2d_3_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv2d_3_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp6assignvariableop_47_adam_batch_normalization_3_gamma_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp5assignvariableop_48_adam_batch_normalization_3_beta_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp'assignvariableop_49_adam_dense_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp%assignvariableop_50_adam_dense_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_conv2d_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp&assignvariableop_52_adam_conv2d_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp4assignvariableop_53_adam_batch_normalization_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp3assignvariableop_54_adam_batch_normalization_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_conv2d_1_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_conv2d_1_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_batch_normalization_1_gamma_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_batch_normalization_1_beta_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_conv2d_2_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_conv2d_2_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_batch_normalization_2_gamma_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp5assignvariableop_62_adam_batch_normalization_2_beta_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_conv2d_3_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_conv2d_3_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_batch_normalization_3_gamma_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_batch_normalization_3_beta_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp'assignvariableop_67_adam_dense_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp%assignvariableop_68_adam_dense_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_69Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_70IdentityIdentity_69:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_70Identity_70:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�.
�
D__inference_triplet_layer_call_and_return_conditional_losses_1185728
anchor_input
positive_input
negative_input*
backbone_1185617:@
backbone_1185619:@
backbone_1185621:@
backbone_1185623:@
backbone_1185625:@
backbone_1185627:@*
backbone_1185629:@@
backbone_1185631:@
backbone_1185633:@
backbone_1185635:@
backbone_1185637:@
backbone_1185639:@*
backbone_1185641:@@
backbone_1185643:@
backbone_1185645:@
backbone_1185647:@
backbone_1185649:@
backbone_1185651:@*
backbone_1185653:@@
backbone_1185655:@
backbone_1185657:@
backbone_1185659:@
backbone_1185661:@
backbone_1185663:@#
backbone_1185665:	�@
backbone_1185667:@
identity�� backbone/StatefulPartitionedCall�"backbone/StatefulPartitionedCall_1�"backbone/StatefulPartitionedCall_2�
 backbone/StatefulPartitionedCallStatefulPartitionedCallanchor_inputbackbone_1185617backbone_1185619backbone_1185621backbone_1185623backbone_1185625backbone_1185627backbone_1185629backbone_1185631backbone_1185633backbone_1185635backbone_1185637backbone_1185639backbone_1185641backbone_1185643backbone_1185645backbone_1185647backbone_1185649backbone_1185651backbone_1185653backbone_1185655backbone_1185657backbone_1185659backbone_1185661backbone_1185663backbone_1185665backbone_1185667*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_backbone_layer_call_and_return_conditional_losses_1184327�
"backbone/StatefulPartitionedCall_1StatefulPartitionedCallnegative_inputbackbone_1185617backbone_1185619backbone_1185621backbone_1185623backbone_1185625backbone_1185627backbone_1185629backbone_1185631backbone_1185633backbone_1185635backbone_1185637backbone_1185639backbone_1185641backbone_1185643backbone_1185645backbone_1185647backbone_1185649backbone_1185651backbone_1185653backbone_1185655backbone_1185657backbone_1185659backbone_1185661backbone_1185663backbone_1185665backbone_1185667*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_backbone_layer_call_and_return_conditional_losses_1184327�
"backbone/StatefulPartitionedCall_2StatefulPartitionedCallpositive_inputbackbone_1185617backbone_1185619backbone_1185621backbone_1185623backbone_1185625backbone_1185627backbone_1185629backbone_1185631backbone_1185633backbone_1185635backbone_1185637backbone_1185639backbone_1185641backbone_1185643backbone_1185645backbone_1185647backbone_1185649backbone_1185651backbone_1185653backbone_1185655backbone_1185657backbone_1185659backbone_1185661backbone_1185663backbone_1185665backbone_1185667*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_backbone_layer_call_and_return_conditional_losses_1184327�
dot/PartitionedCallPartitionedCall)backbone/StatefulPartitionedCall:output:0+backbone/StatefulPartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dot_layer_call_and_return_conditional_losses_1185203�
dot_1/PartitionedCallPartitionedCall)backbone/StatefulPartitionedCall:output:0+backbone/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dot_1_layer_call_and_return_conditional_losses_1185231�
concatenate/PartitionedCallPartitionedCalldot/PartitionedCall:output:0dot_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1185240s
IdentityIdentity$concatenate/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^backbone/StatefulPartitionedCall#^backbone/StatefulPartitionedCall_1#^backbone/StatefulPartitionedCall_2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������~:����������~:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 backbone/StatefulPartitionedCall backbone/StatefulPartitionedCall2H
"backbone/StatefulPartitionedCall_1"backbone/StatefulPartitionedCall_12H
"backbone/StatefulPartitionedCall_2"backbone/StatefulPartitionedCall_2:^ Z
0
_output_shapes
:����������~
&
_user_specified_nameanchor_input:`\
0
_output_shapes
:����������~
(
_user_specified_namepositive_input:`\
0
_output_shapes
:����������~
(
_user_specified_namenegative_input
�
l
@__inference_dot_layer_call_and_return_conditional_losses_1186955
inputs_0
inputs_1
identityY
l2_normalize/SquareSquareinputs_0*
T0*'
_output_shapes
:���������@d
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������g
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������g
l2_normalizeMulinputs_0l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@[
l2_normalize_1/SquareSquareinputs_1*
T0*'
_output_shapes
:���������@f
$l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize_1/SumSuml2_normalize_1/Square:y:0-l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(]
l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize_1/MaximumMaximuml2_normalize_1/Sum:output:0!l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������k
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:���������k
l2_normalize_1Mulinputs_1l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:���������@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :y

ExpandDims
ExpandDimsl2_normalize:z:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������@R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsl2_normalize_1:z:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:���������@y
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:���������D
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:l
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
X
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs/1
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1184105

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:����������~@:@:@:@:@:*
epsilon%o�:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:����������~@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������~@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:����������~@
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1187514

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
)__inference_triplet_layer_call_fn_1185970
inputs_0
inputs_1
inputs_2!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	�@

unknown_24:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_triplet_layer_call_and_return_conditional_losses_1185243o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������~:����������~:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:����������~
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:����������~
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:����������~
"
_user_specified_name
inputs/2
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_1184308

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1187107

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
j
@__inference_dot_layer_call_and_return_conditional_losses_1185203

inputs
inputs_1
identityW
l2_normalize/SquareSquareinputs*
T0*'
_output_shapes
:���������@d
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������g
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������e
l2_normalizeMulinputsl2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@[
l2_normalize_1/SquareSquareinputs_1*
T0*'
_output_shapes
:���������@f
$l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize_1/SumSuml2_normalize_1/Square:y:0-l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(]
l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize_1/MaximumMaximuml2_normalize_1/Sum:output:0!l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������k
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:���������k
l2_normalize_1Mulinputs_1l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:���������@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :y

ExpandDims
ExpandDimsl2_normalize:z:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������@R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsl2_normalize_1:z:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:���������@y
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:���������D
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:l
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
X
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1184273

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1184447

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
f
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1187168

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_1_layer_call_fn_1187244

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1184577w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@?@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@?@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@?@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1187435

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1183954

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
f
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1184126

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������@?@*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������@?@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������~@:X T
0
_output_shapes
:����������~@
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_1_layer_call_fn_1187336

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1184182h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@?@:W S
/
_output_shapes
:���������@?@
 
_user_specified_nameinputs
��
�0
D__inference_triplet_layer_call_and_return_conditional_losses_1186318
inputs_0
inputs_1
inputs_2H
.backbone_conv2d_conv2d_readvariableop_resource:@=
/backbone_conv2d_biasadd_readvariableop_resource:@B
4backbone_batch_normalization_readvariableop_resource:@D
6backbone_batch_normalization_readvariableop_1_resource:@S
Ebackbone_batch_normalization_fusedbatchnormv3_readvariableop_resource:@U
Gbackbone_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@J
0backbone_conv2d_1_conv2d_readvariableop_resource:@@?
1backbone_conv2d_1_biasadd_readvariableop_resource:@D
6backbone_batch_normalization_1_readvariableop_resource:@F
8backbone_batch_normalization_1_readvariableop_1_resource:@U
Gbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@W
Ibackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@J
0backbone_conv2d_2_conv2d_readvariableop_resource:@@?
1backbone_conv2d_2_biasadd_readvariableop_resource:@D
6backbone_batch_normalization_2_readvariableop_resource:@F
8backbone_batch_normalization_2_readvariableop_1_resource:@U
Gbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@W
Ibackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@J
0backbone_conv2d_3_conv2d_readvariableop_resource:@@?
1backbone_conv2d_3_biasadd_readvariableop_resource:@D
6backbone_batch_normalization_3_readvariableop_resource:@F
8backbone_batch_normalization_3_readvariableop_1_resource:@U
Gbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:@W
Ibackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:@@
-backbone_dense_matmul_readvariableop_resource:	�@<
.backbone_dense_biasadd_readvariableop_resource:@
identity��<backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp�>backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�>backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp�@backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1�>backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp�@backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_1�+backbone/batch_normalization/ReadVariableOp�-backbone/batch_normalization/ReadVariableOp_1�-backbone/batch_normalization/ReadVariableOp_2�-backbone/batch_normalization/ReadVariableOp_3�-backbone/batch_normalization/ReadVariableOp_4�-backbone/batch_normalization/ReadVariableOp_5�>backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�@backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�@backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp�Bbackbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1�@backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp�Bbackbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_1�-backbone/batch_normalization_1/ReadVariableOp�/backbone/batch_normalization_1/ReadVariableOp_1�/backbone/batch_normalization_1/ReadVariableOp_2�/backbone/batch_normalization_1/ReadVariableOp_3�/backbone/batch_normalization_1/ReadVariableOp_4�/backbone/batch_normalization_1/ReadVariableOp_5�>backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�@backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�@backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp�Bbackbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1�@backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp�Bbackbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_1�-backbone/batch_normalization_2/ReadVariableOp�/backbone/batch_normalization_2/ReadVariableOp_1�/backbone/batch_normalization_2/ReadVariableOp_2�/backbone/batch_normalization_2/ReadVariableOp_3�/backbone/batch_normalization_2/ReadVariableOp_4�/backbone/batch_normalization_2/ReadVariableOp_5�>backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�@backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�@backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp�Bbackbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1�@backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp�Bbackbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_1�-backbone/batch_normalization_3/ReadVariableOp�/backbone/batch_normalization_3/ReadVariableOp_1�/backbone/batch_normalization_3/ReadVariableOp_2�/backbone/batch_normalization_3/ReadVariableOp_3�/backbone/batch_normalization_3/ReadVariableOp_4�/backbone/batch_normalization_3/ReadVariableOp_5�&backbone/conv2d/BiasAdd/ReadVariableOp�(backbone/conv2d/BiasAdd_1/ReadVariableOp�(backbone/conv2d/BiasAdd_2/ReadVariableOp�%backbone/conv2d/Conv2D/ReadVariableOp�'backbone/conv2d/Conv2D_1/ReadVariableOp�'backbone/conv2d/Conv2D_2/ReadVariableOp�(backbone/conv2d_1/BiasAdd/ReadVariableOp�*backbone/conv2d_1/BiasAdd_1/ReadVariableOp�*backbone/conv2d_1/BiasAdd_2/ReadVariableOp�'backbone/conv2d_1/Conv2D/ReadVariableOp�)backbone/conv2d_1/Conv2D_1/ReadVariableOp�)backbone/conv2d_1/Conv2D_2/ReadVariableOp�(backbone/conv2d_2/BiasAdd/ReadVariableOp�*backbone/conv2d_2/BiasAdd_1/ReadVariableOp�*backbone/conv2d_2/BiasAdd_2/ReadVariableOp�'backbone/conv2d_2/Conv2D/ReadVariableOp�)backbone/conv2d_2/Conv2D_1/ReadVariableOp�)backbone/conv2d_2/Conv2D_2/ReadVariableOp�(backbone/conv2d_3/BiasAdd/ReadVariableOp�*backbone/conv2d_3/BiasAdd_1/ReadVariableOp�*backbone/conv2d_3/BiasAdd_2/ReadVariableOp�'backbone/conv2d_3/Conv2D/ReadVariableOp�)backbone/conv2d_3/Conv2D_1/ReadVariableOp�)backbone/conv2d_3/Conv2D_2/ReadVariableOp�%backbone/dense/BiasAdd/ReadVariableOp�'backbone/dense/BiasAdd_1/ReadVariableOp�'backbone/dense/BiasAdd_2/ReadVariableOp�$backbone/dense/MatMul/ReadVariableOp�&backbone/dense/MatMul_1/ReadVariableOp�&backbone/dense/MatMul_2/ReadVariableOp�
%backbone/conv2d/Conv2D/ReadVariableOpReadVariableOp.backbone_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
backbone/conv2d/Conv2DConv2Dinputs_0-backbone/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@*
paddingSAME*
strides
�
&backbone/conv2d/BiasAdd/ReadVariableOpReadVariableOp/backbone_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d/BiasAddBiasAddbackbone/conv2d/Conv2D:output:0.backbone/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@�
+backbone/batch_normalization/ReadVariableOpReadVariableOp4backbone_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
-backbone/batch_normalization/ReadVariableOp_1ReadVariableOp6backbone_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
<backbone/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpEbackbone_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
>backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGbackbone_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
-backbone/batch_normalization/FusedBatchNormV3FusedBatchNormV3 backbone/conv2d/BiasAdd:output:03backbone/batch_normalization/ReadVariableOp:value:05backbone/batch_normalization/ReadVariableOp_1:value:0Dbackbone/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Fbackbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:����������~@:@:@:@:@:*
epsilon%o�:*
is_training( �
backbone/activation/ReluRelu1backbone/batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������~@�
backbone/max_pooling2d/MaxPoolMaxPool&backbone/activation/Relu:activations:0*/
_output_shapes
:���������@?@*
ksize
*
paddingVALID*
strides
�
'backbone/conv2d_1/Conv2D/ReadVariableOpReadVariableOp0backbone_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_1/Conv2DConv2D'backbone/max_pooling2d/MaxPool:output:0/backbone/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@*
paddingSAME*
strides
�
(backbone/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp1backbone_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_1/BiasAddBiasAdd!backbone/conv2d_1/Conv2D:output:00backbone/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@�
-backbone/batch_normalization_1/ReadVariableOpReadVariableOp6backbone_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_1/ReadVariableOp_1ReadVariableOp8backbone_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
>backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpGbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3"backbone/conv2d_1/BiasAdd:output:05backbone/batch_normalization_1/ReadVariableOp:value:07backbone/batch_normalization_1/ReadVariableOp_1:value:0Fbackbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Hbackbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@?@:@:@:@:@:*
epsilon%o�:*
is_training( �
backbone/activation_1/ReluRelu3backbone/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@?@�
 backbone/max_pooling2d_1/MaxPoolMaxPool(backbone/activation_1/Relu:activations:0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
�
'backbone/conv2d_2/Conv2D/ReadVariableOpReadVariableOp0backbone_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_2/Conv2DConv2D)backbone/max_pooling2d_1/MaxPool:output:0/backbone/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
�
(backbone/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp1backbone_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_2/BiasAddBiasAdd!backbone/conv2d_2/Conv2D:output:00backbone/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @�
-backbone/batch_normalization_2/ReadVariableOpReadVariableOp6backbone_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_2/ReadVariableOp_1ReadVariableOp8backbone_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
>backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpGbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3"backbone/conv2d_2/BiasAdd:output:05backbone/batch_normalization_2/ReadVariableOp:value:07backbone/batch_normalization_2/ReadVariableOp_1:value:0Fbackbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Hbackbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� @:@:@:@:@:*
epsilon%o�:*
is_training( �
backbone/activation_2/ReluRelu3backbone/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� @�
 backbone/max_pooling2d_2/MaxPoolMaxPool(backbone/activation_2/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
'backbone/conv2d_3/Conv2D/ReadVariableOpReadVariableOp0backbone_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_3/Conv2DConv2D)backbone/max_pooling2d_2/MaxPool:output:0/backbone/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
(backbone/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp1backbone_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_3/BiasAddBiasAdd!backbone/conv2d_3/Conv2D:output:00backbone/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
-backbone/batch_normalization_3/ReadVariableOpReadVariableOp6backbone_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_3/ReadVariableOp_1ReadVariableOp8backbone_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
>backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpGbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3"backbone/conv2d_3/BiasAdd:output:05backbone/batch_normalization_3/ReadVariableOp:value:07backbone/batch_normalization_3/ReadVariableOp_1:value:0Fbackbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Hbackbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
backbone/activation_3/ReluRelu3backbone/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@�
 backbone/max_pooling2d_3/MaxPoolMaxPool(backbone/activation_3/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
 backbone/max_pooling2d_4/MaxPoolMaxPool)backbone/max_pooling2d_3/MaxPool:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
g
backbone/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
backbone/flatten/ReshapeReshape)backbone/max_pooling2d_4/MaxPool:output:0backbone/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
$backbone/dense/MatMul/ReadVariableOpReadVariableOp-backbone_dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
backbone/dense/MatMulMatMul!backbone/flatten/Reshape:output:0,backbone/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
%backbone/dense/BiasAdd/ReadVariableOpReadVariableOp.backbone_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/dense/BiasAddBiasAddbackbone/dense/MatMul:product:0-backbone/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'backbone/conv2d/Conv2D_1/ReadVariableOpReadVariableOp.backbone_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
backbone/conv2d/Conv2D_1Conv2Dinputs_2/backbone/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@*
paddingSAME*
strides
�
(backbone/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp/backbone_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d/BiasAdd_1BiasAdd!backbone/conv2d/Conv2D_1:output:00backbone/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@�
-backbone/batch_normalization/ReadVariableOp_2ReadVariableOp4backbone_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
-backbone/batch_normalization/ReadVariableOp_3ReadVariableOp6backbone_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
>backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEbackbone_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGbackbone_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization/FusedBatchNormV3_1FusedBatchNormV3"backbone/conv2d/BiasAdd_1:output:05backbone/batch_normalization/ReadVariableOp_2:value:05backbone/batch_normalization/ReadVariableOp_3:value:0Fbackbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp:value:0Hbackbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:����������~@:@:@:@:@:*
epsilon%o�:*
is_training( �
backbone/activation/Relu_1Relu3backbone/batch_normalization/FusedBatchNormV3_1:y:0*
T0*0
_output_shapes
:����������~@�
 backbone/max_pooling2d/MaxPool_1MaxPool(backbone/activation/Relu_1:activations:0*/
_output_shapes
:���������@?@*
ksize
*
paddingVALID*
strides
�
)backbone/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp0backbone_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_1/Conv2D_1Conv2D)backbone/max_pooling2d/MaxPool_1:output:01backbone/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@*
paddingSAME*
strides
�
*backbone/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp1backbone_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_1/BiasAdd_1BiasAdd#backbone/conv2d_1/Conv2D_1:output:02backbone/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@�
/backbone/batch_normalization_1/ReadVariableOp_2ReadVariableOp6backbone_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_1/ReadVariableOp_3ReadVariableOp8backbone_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpReadVariableOpGbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Bbackbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
1backbone/batch_normalization_1/FusedBatchNormV3_1FusedBatchNormV3$backbone/conv2d_1/BiasAdd_1:output:07backbone/batch_normalization_1/ReadVariableOp_2:value:07backbone/batch_normalization_1/ReadVariableOp_3:value:0Hbackbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp:value:0Jbackbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@?@:@:@:@:@:*
epsilon%o�:*
is_training( �
backbone/activation_1/Relu_1Relu5backbone/batch_normalization_1/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:���������@?@�
"backbone/max_pooling2d_1/MaxPool_1MaxPool*backbone/activation_1/Relu_1:activations:0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
�
)backbone/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOp0backbone_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_2/Conv2D_1Conv2D+backbone/max_pooling2d_1/MaxPool_1:output:01backbone/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
�
*backbone/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOp1backbone_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_2/BiasAdd_1BiasAdd#backbone/conv2d_2/Conv2D_1:output:02backbone/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @�
/backbone/batch_normalization_2/ReadVariableOp_2ReadVariableOp6backbone_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_2/ReadVariableOp_3ReadVariableOp8backbone_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpReadVariableOpGbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Bbackbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
1backbone/batch_normalization_2/FusedBatchNormV3_1FusedBatchNormV3$backbone/conv2d_2/BiasAdd_1:output:07backbone/batch_normalization_2/ReadVariableOp_2:value:07backbone/batch_normalization_2/ReadVariableOp_3:value:0Hbackbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp:value:0Jbackbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� @:@:@:@:@:*
epsilon%o�:*
is_training( �
backbone/activation_2/Relu_1Relu5backbone/batch_normalization_2/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:��������� @�
"backbone/max_pooling2d_2/MaxPool_1MaxPool*backbone/activation_2/Relu_1:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
)backbone/conv2d_3/Conv2D_1/ReadVariableOpReadVariableOp0backbone_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_3/Conv2D_1Conv2D+backbone/max_pooling2d_2/MaxPool_1:output:01backbone/conv2d_3/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
*backbone/conv2d_3/BiasAdd_1/ReadVariableOpReadVariableOp1backbone_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_3/BiasAdd_1BiasAdd#backbone/conv2d_3/Conv2D_1:output:02backbone/conv2d_3/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
/backbone/batch_normalization_3/ReadVariableOp_2ReadVariableOp6backbone_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_3/ReadVariableOp_3ReadVariableOp8backbone_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpReadVariableOpGbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Bbackbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
1backbone/batch_normalization_3/FusedBatchNormV3_1FusedBatchNormV3$backbone/conv2d_3/BiasAdd_1:output:07backbone/batch_normalization_3/ReadVariableOp_2:value:07backbone/batch_normalization_3/ReadVariableOp_3:value:0Hbackbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp:value:0Jbackbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
backbone/activation_3/Relu_1Relu5backbone/batch_normalization_3/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:���������@�
"backbone/max_pooling2d_3/MaxPool_1MaxPool*backbone/activation_3/Relu_1:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
"backbone/max_pooling2d_4/MaxPool_1MaxPool+backbone/max_pooling2d_3/MaxPool_1:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
i
backbone/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"����   �
backbone/flatten/Reshape_1Reshape+backbone/max_pooling2d_4/MaxPool_1:output:0!backbone/flatten/Const_1:output:0*
T0*(
_output_shapes
:�����������
&backbone/dense/MatMul_1/ReadVariableOpReadVariableOp-backbone_dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
backbone/dense/MatMul_1MatMul#backbone/flatten/Reshape_1:output:0.backbone/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'backbone/dense/BiasAdd_1/ReadVariableOpReadVariableOp.backbone_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/dense/BiasAdd_1BiasAdd!backbone/dense/MatMul_1:product:0/backbone/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'backbone/conv2d/Conv2D_2/ReadVariableOpReadVariableOp.backbone_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
backbone/conv2d/Conv2D_2Conv2Dinputs_1/backbone/conv2d/Conv2D_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@*
paddingSAME*
strides
�
(backbone/conv2d/BiasAdd_2/ReadVariableOpReadVariableOp/backbone_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d/BiasAdd_2BiasAdd!backbone/conv2d/Conv2D_2:output:00backbone/conv2d/BiasAdd_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������~@�
-backbone/batch_normalization/ReadVariableOp_4ReadVariableOp4backbone_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
-backbone/batch_normalization/ReadVariableOp_5ReadVariableOp6backbone_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
>backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOpReadVariableOpEbackbone_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_1ReadVariableOpGbackbone_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization/FusedBatchNormV3_2FusedBatchNormV3"backbone/conv2d/BiasAdd_2:output:05backbone/batch_normalization/ReadVariableOp_4:value:05backbone/batch_normalization/ReadVariableOp_5:value:0Fbackbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp:value:0Hbackbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:����������~@:@:@:@:@:*
epsilon%o�:*
is_training( �
backbone/activation/Relu_2Relu3backbone/batch_normalization/FusedBatchNormV3_2:y:0*
T0*0
_output_shapes
:����������~@�
 backbone/max_pooling2d/MaxPool_2MaxPool(backbone/activation/Relu_2:activations:0*/
_output_shapes
:���������@?@*
ksize
*
paddingVALID*
strides
�
)backbone/conv2d_1/Conv2D_2/ReadVariableOpReadVariableOp0backbone_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_1/Conv2D_2Conv2D)backbone/max_pooling2d/MaxPool_2:output:01backbone/conv2d_1/Conv2D_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@*
paddingSAME*
strides
�
*backbone/conv2d_1/BiasAdd_2/ReadVariableOpReadVariableOp1backbone_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_1/BiasAdd_2BiasAdd#backbone/conv2d_1/Conv2D_2:output:02backbone/conv2d_1/BiasAdd_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@?@�
/backbone/batch_normalization_1/ReadVariableOp_4ReadVariableOp6backbone_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_1/ReadVariableOp_5ReadVariableOp8backbone_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOpReadVariableOpGbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Bbackbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
1backbone/batch_normalization_1/FusedBatchNormV3_2FusedBatchNormV3$backbone/conv2d_1/BiasAdd_2:output:07backbone/batch_normalization_1/ReadVariableOp_4:value:07backbone/batch_normalization_1/ReadVariableOp_5:value:0Hbackbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp:value:0Jbackbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@?@:@:@:@:@:*
epsilon%o�:*
is_training( �
backbone/activation_1/Relu_2Relu5backbone/batch_normalization_1/FusedBatchNormV3_2:y:0*
T0*/
_output_shapes
:���������@?@�
"backbone/max_pooling2d_1/MaxPool_2MaxPool*backbone/activation_1/Relu_2:activations:0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
�
)backbone/conv2d_2/Conv2D_2/ReadVariableOpReadVariableOp0backbone_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_2/Conv2D_2Conv2D+backbone/max_pooling2d_1/MaxPool_2:output:01backbone/conv2d_2/Conv2D_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @*
paddingSAME*
strides
�
*backbone/conv2d_2/BiasAdd_2/ReadVariableOpReadVariableOp1backbone_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_2/BiasAdd_2BiasAdd#backbone/conv2d_2/Conv2D_2:output:02backbone/conv2d_2/BiasAdd_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� @�
/backbone/batch_normalization_2/ReadVariableOp_4ReadVariableOp6backbone_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_2/ReadVariableOp_5ReadVariableOp8backbone_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOpReadVariableOpGbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Bbackbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
1backbone/batch_normalization_2/FusedBatchNormV3_2FusedBatchNormV3$backbone/conv2d_2/BiasAdd_2:output:07backbone/batch_normalization_2/ReadVariableOp_4:value:07backbone/batch_normalization_2/ReadVariableOp_5:value:0Hbackbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp:value:0Jbackbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� @:@:@:@:@:*
epsilon%o�:*
is_training( �
backbone/activation_2/Relu_2Relu5backbone/batch_normalization_2/FusedBatchNormV3_2:y:0*
T0*/
_output_shapes
:��������� @�
"backbone/max_pooling2d_2/MaxPool_2MaxPool*backbone/activation_2/Relu_2:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
)backbone/conv2d_3/Conv2D_2/ReadVariableOpReadVariableOp0backbone_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
backbone/conv2d_3/Conv2D_2Conv2D+backbone/max_pooling2d_2/MaxPool_2:output:01backbone/conv2d_3/Conv2D_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
*backbone/conv2d_3/BiasAdd_2/ReadVariableOpReadVariableOp1backbone_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/conv2d_3/BiasAdd_2BiasAdd#backbone/conv2d_3/Conv2D_2:output:02backbone/conv2d_3/BiasAdd_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
/backbone/batch_normalization_3/ReadVariableOp_4ReadVariableOp6backbone_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0�
/backbone/batch_normalization_3/ReadVariableOp_5ReadVariableOp8backbone_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
@backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOpReadVariableOpGbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Bbackbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_1ReadVariableOpIbackbone_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
1backbone/batch_normalization_3/FusedBatchNormV3_2FusedBatchNormV3$backbone/conv2d_3/BiasAdd_2:output:07backbone/batch_normalization_3/ReadVariableOp_4:value:07backbone/batch_normalization_3/ReadVariableOp_5:value:0Hbackbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp:value:0Jbackbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
backbone/activation_3/Relu_2Relu5backbone/batch_normalization_3/FusedBatchNormV3_2:y:0*
T0*/
_output_shapes
:���������@�
"backbone/max_pooling2d_3/MaxPool_2MaxPool*backbone/activation_3/Relu_2:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
"backbone/max_pooling2d_4/MaxPool_2MaxPool+backbone/max_pooling2d_3/MaxPool_2:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
i
backbone/flatten/Const_2Const*
_output_shapes
:*
dtype0*
valueB"����   �
backbone/flatten/Reshape_2Reshape+backbone/max_pooling2d_4/MaxPool_2:output:0!backbone/flatten/Const_2:output:0*
T0*(
_output_shapes
:�����������
&backbone/dense/MatMul_2/ReadVariableOpReadVariableOp-backbone_dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
backbone/dense/MatMul_2MatMul#backbone/flatten/Reshape_2:output:0.backbone/dense/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'backbone/dense/BiasAdd_2/ReadVariableOpReadVariableOp.backbone_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
backbone/dense/BiasAdd_2BiasAdd!backbone/dense/MatMul_2:product:0/backbone/dense/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@t
dot/l2_normalize/SquareSquarebackbone/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@h
&dot/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
dot/l2_normalize/SumSumdot/l2_normalize/Square:y:0/dot/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(_
dot/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
dot/l2_normalize/MaximumMaximumdot/l2_normalize/Sum:output:0#dot/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������o
dot/l2_normalize/RsqrtRsqrtdot/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
dot/l2_normalizeMulbackbone/dense/BiasAdd:output:0dot/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@x
dot/l2_normalize_1/SquareSquare!backbone/dense/BiasAdd_2:output:0*
T0*'
_output_shapes
:���������@j
(dot/l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
dot/l2_normalize_1/SumSumdot/l2_normalize_1/Square:y:01dot/l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(a
dot/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
dot/l2_normalize_1/MaximumMaximumdot/l2_normalize_1/Sum:output:0%dot/l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������s
dot/l2_normalize_1/RsqrtRsqrtdot/l2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:����������
dot/l2_normalize_1Mul!backbone/dense/BiasAdd_2:output:0dot/l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:���������@T
dot/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot/ExpandDims
ExpandDimsdot/l2_normalize:z:0dot/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������@V
dot/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot/ExpandDims_1
ExpandDimsdot/l2_normalize_1:z:0dot/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:���������@�

dot/MatMulBatchMatMulV2dot/ExpandDims:output:0dot/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������L
	dot/ShapeShapedot/MatMul:output:0*
T0*
_output_shapes
:t
dot/SqueezeSqueezedot/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
v
dot_1/l2_normalize/SquareSquarebackbone/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@j
(dot_1/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
dot_1/l2_normalize/SumSumdot_1/l2_normalize/Square:y:01dot_1/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(a
dot_1/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
dot_1/l2_normalize/MaximumMaximumdot_1/l2_normalize/Sum:output:0%dot_1/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������s
dot_1/l2_normalize/RsqrtRsqrtdot_1/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
dot_1/l2_normalizeMulbackbone/dense/BiasAdd:output:0dot_1/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@z
dot_1/l2_normalize_1/SquareSquare!backbone/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:���������@l
*dot_1/l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
dot_1/l2_normalize_1/SumSumdot_1/l2_normalize_1/Square:y:03dot_1/l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(c
dot_1/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
dot_1/l2_normalize_1/MaximumMaximum!dot_1/l2_normalize_1/Sum:output:0'dot_1/l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������w
dot_1/l2_normalize_1/RsqrtRsqrt dot_1/l2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:����������
dot_1/l2_normalize_1Mul!backbone/dense/BiasAdd_1:output:0dot_1/l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:���������@V
dot_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot_1/ExpandDims
ExpandDimsdot_1/l2_normalize:z:0dot_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������@X
dot_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot_1/ExpandDims_1
ExpandDimsdot_1/l2_normalize_1:z:0dot_1/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:���������@�
dot_1/MatMulBatchMatMulV2dot_1/ExpandDims:output:0dot_1/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������P
dot_1/ShapeShapedot_1/MatMul:output:0*
T0*
_output_shapes
:x
dot_1/SqueezeSqueezedot_1/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2dot/Squeeze:output:0dot_1/Squeeze:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������j
IdentityIdentityconcatenate/concat:output:0^NoOp*
T0*'
_output_shapes
:���������� 
NoOpNoOp=^backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp?^backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?^backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOpA^backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1?^backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOpA^backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_1,^backbone/batch_normalization/ReadVariableOp.^backbone/batch_normalization/ReadVariableOp_1.^backbone/batch_normalization/ReadVariableOp_2.^backbone/batch_normalization/ReadVariableOp_3.^backbone/batch_normalization/ReadVariableOp_4.^backbone/batch_normalization/ReadVariableOp_5?^backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOpA^backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1A^backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpC^backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1A^backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOpC^backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_1.^backbone/batch_normalization_1/ReadVariableOp0^backbone/batch_normalization_1/ReadVariableOp_10^backbone/batch_normalization_1/ReadVariableOp_20^backbone/batch_normalization_1/ReadVariableOp_30^backbone/batch_normalization_1/ReadVariableOp_40^backbone/batch_normalization_1/ReadVariableOp_5?^backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOpA^backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1A^backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpC^backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1A^backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOpC^backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_1.^backbone/batch_normalization_2/ReadVariableOp0^backbone/batch_normalization_2/ReadVariableOp_10^backbone/batch_normalization_2/ReadVariableOp_20^backbone/batch_normalization_2/ReadVariableOp_30^backbone/batch_normalization_2/ReadVariableOp_40^backbone/batch_normalization_2/ReadVariableOp_5?^backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOpA^backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1A^backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpC^backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1A^backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOpC^backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_1.^backbone/batch_normalization_3/ReadVariableOp0^backbone/batch_normalization_3/ReadVariableOp_10^backbone/batch_normalization_3/ReadVariableOp_20^backbone/batch_normalization_3/ReadVariableOp_30^backbone/batch_normalization_3/ReadVariableOp_40^backbone/batch_normalization_3/ReadVariableOp_5'^backbone/conv2d/BiasAdd/ReadVariableOp)^backbone/conv2d/BiasAdd_1/ReadVariableOp)^backbone/conv2d/BiasAdd_2/ReadVariableOp&^backbone/conv2d/Conv2D/ReadVariableOp(^backbone/conv2d/Conv2D_1/ReadVariableOp(^backbone/conv2d/Conv2D_2/ReadVariableOp)^backbone/conv2d_1/BiasAdd/ReadVariableOp+^backbone/conv2d_1/BiasAdd_1/ReadVariableOp+^backbone/conv2d_1/BiasAdd_2/ReadVariableOp(^backbone/conv2d_1/Conv2D/ReadVariableOp*^backbone/conv2d_1/Conv2D_1/ReadVariableOp*^backbone/conv2d_1/Conv2D_2/ReadVariableOp)^backbone/conv2d_2/BiasAdd/ReadVariableOp+^backbone/conv2d_2/BiasAdd_1/ReadVariableOp+^backbone/conv2d_2/BiasAdd_2/ReadVariableOp(^backbone/conv2d_2/Conv2D/ReadVariableOp*^backbone/conv2d_2/Conv2D_1/ReadVariableOp*^backbone/conv2d_2/Conv2D_2/ReadVariableOp)^backbone/conv2d_3/BiasAdd/ReadVariableOp+^backbone/conv2d_3/BiasAdd_1/ReadVariableOp+^backbone/conv2d_3/BiasAdd_2/ReadVariableOp(^backbone/conv2d_3/Conv2D/ReadVariableOp*^backbone/conv2d_3/Conv2D_1/ReadVariableOp*^backbone/conv2d_3/Conv2D_2/ReadVariableOp&^backbone/dense/BiasAdd/ReadVariableOp(^backbone/dense/BiasAdd_1/ReadVariableOp(^backbone/dense/BiasAdd_2/ReadVariableOp%^backbone/dense/MatMul/ReadVariableOp'^backbone/dense/MatMul_1/ReadVariableOp'^backbone/dense/MatMul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������~:����������~:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 2|
<backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp<backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp2�
>backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_1>backbone/batch_normalization/FusedBatchNormV3/ReadVariableOp_12�
>backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp>backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp2�
@backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1@backbone/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_12�
>backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp>backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp2�
@backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_1@backbone/batch_normalization/FusedBatchNormV3_2/ReadVariableOp_12Z
+backbone/batch_normalization/ReadVariableOp+backbone/batch_normalization/ReadVariableOp2^
-backbone/batch_normalization/ReadVariableOp_1-backbone/batch_normalization/ReadVariableOp_12^
-backbone/batch_normalization/ReadVariableOp_2-backbone/batch_normalization/ReadVariableOp_22^
-backbone/batch_normalization/ReadVariableOp_3-backbone/batch_normalization/ReadVariableOp_32^
-backbone/batch_normalization/ReadVariableOp_4-backbone/batch_normalization/ReadVariableOp_42^
-backbone/batch_normalization/ReadVariableOp_5-backbone/batch_normalization/ReadVariableOp_52�
>backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2�
@backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1@backbone/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12�
@backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp@backbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp2�
Bbackbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1Bbackbone/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_12�
@backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp@backbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp2�
Bbackbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_1Bbackbone/batch_normalization_1/FusedBatchNormV3_2/ReadVariableOp_12^
-backbone/batch_normalization_1/ReadVariableOp-backbone/batch_normalization_1/ReadVariableOp2b
/backbone/batch_normalization_1/ReadVariableOp_1/backbone/batch_normalization_1/ReadVariableOp_12b
/backbone/batch_normalization_1/ReadVariableOp_2/backbone/batch_normalization_1/ReadVariableOp_22b
/backbone/batch_normalization_1/ReadVariableOp_3/backbone/batch_normalization_1/ReadVariableOp_32b
/backbone/batch_normalization_1/ReadVariableOp_4/backbone/batch_normalization_1/ReadVariableOp_42b
/backbone/batch_normalization_1/ReadVariableOp_5/backbone/batch_normalization_1/ReadVariableOp_52�
>backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp>backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2�
@backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1@backbone/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12�
@backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp@backbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp2�
Bbackbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1Bbackbone/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_12�
@backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp@backbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp2�
Bbackbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_1Bbackbone/batch_normalization_2/FusedBatchNormV3_2/ReadVariableOp_12^
-backbone/batch_normalization_2/ReadVariableOp-backbone/batch_normalization_2/ReadVariableOp2b
/backbone/batch_normalization_2/ReadVariableOp_1/backbone/batch_normalization_2/ReadVariableOp_12b
/backbone/batch_normalization_2/ReadVariableOp_2/backbone/batch_normalization_2/ReadVariableOp_22b
/backbone/batch_normalization_2/ReadVariableOp_3/backbone/batch_normalization_2/ReadVariableOp_32b
/backbone/batch_normalization_2/ReadVariableOp_4/backbone/batch_normalization_2/ReadVariableOp_42b
/backbone/batch_normalization_2/ReadVariableOp_5/backbone/batch_normalization_2/ReadVariableOp_52�
>backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp>backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
@backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1@backbone/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12�
@backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp@backbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp2�
Bbackbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1Bbackbone/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_12�
@backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp@backbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp2�
Bbackbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_1Bbackbone/batch_normalization_3/FusedBatchNormV3_2/ReadVariableOp_12^
-backbone/batch_normalization_3/ReadVariableOp-backbone/batch_normalization_3/ReadVariableOp2b
/backbone/batch_normalization_3/ReadVariableOp_1/backbone/batch_normalization_3/ReadVariableOp_12b
/backbone/batch_normalization_3/ReadVariableOp_2/backbone/batch_normalization_3/ReadVariableOp_22b
/backbone/batch_normalization_3/ReadVariableOp_3/backbone/batch_normalization_3/ReadVariableOp_32b
/backbone/batch_normalization_3/ReadVariableOp_4/backbone/batch_normalization_3/ReadVariableOp_42b
/backbone/batch_normalization_3/ReadVariableOp_5/backbone/batch_normalization_3/ReadVariableOp_52P
&backbone/conv2d/BiasAdd/ReadVariableOp&backbone/conv2d/BiasAdd/ReadVariableOp2T
(backbone/conv2d/BiasAdd_1/ReadVariableOp(backbone/conv2d/BiasAdd_1/ReadVariableOp2T
(backbone/conv2d/BiasAdd_2/ReadVariableOp(backbone/conv2d/BiasAdd_2/ReadVariableOp2N
%backbone/conv2d/Conv2D/ReadVariableOp%backbone/conv2d/Conv2D/ReadVariableOp2R
'backbone/conv2d/Conv2D_1/ReadVariableOp'backbone/conv2d/Conv2D_1/ReadVariableOp2R
'backbone/conv2d/Conv2D_2/ReadVariableOp'backbone/conv2d/Conv2D_2/ReadVariableOp2T
(backbone/conv2d_1/BiasAdd/ReadVariableOp(backbone/conv2d_1/BiasAdd/ReadVariableOp2X
*backbone/conv2d_1/BiasAdd_1/ReadVariableOp*backbone/conv2d_1/BiasAdd_1/ReadVariableOp2X
*backbone/conv2d_1/BiasAdd_2/ReadVariableOp*backbone/conv2d_1/BiasAdd_2/ReadVariableOp2R
'backbone/conv2d_1/Conv2D/ReadVariableOp'backbone/conv2d_1/Conv2D/ReadVariableOp2V
)backbone/conv2d_1/Conv2D_1/ReadVariableOp)backbone/conv2d_1/Conv2D_1/ReadVariableOp2V
)backbone/conv2d_1/Conv2D_2/ReadVariableOp)backbone/conv2d_1/Conv2D_2/ReadVariableOp2T
(backbone/conv2d_2/BiasAdd/ReadVariableOp(backbone/conv2d_2/BiasAdd/ReadVariableOp2X
*backbone/conv2d_2/BiasAdd_1/ReadVariableOp*backbone/conv2d_2/BiasAdd_1/ReadVariableOp2X
*backbone/conv2d_2/BiasAdd_2/ReadVariableOp*backbone/conv2d_2/BiasAdd_2/ReadVariableOp2R
'backbone/conv2d_2/Conv2D/ReadVariableOp'backbone/conv2d_2/Conv2D/ReadVariableOp2V
)backbone/conv2d_2/Conv2D_1/ReadVariableOp)backbone/conv2d_2/Conv2D_1/ReadVariableOp2V
)backbone/conv2d_2/Conv2D_2/ReadVariableOp)backbone/conv2d_2/Conv2D_2/ReadVariableOp2T
(backbone/conv2d_3/BiasAdd/ReadVariableOp(backbone/conv2d_3/BiasAdd/ReadVariableOp2X
*backbone/conv2d_3/BiasAdd_1/ReadVariableOp*backbone/conv2d_3/BiasAdd_1/ReadVariableOp2X
*backbone/conv2d_3/BiasAdd_2/ReadVariableOp*backbone/conv2d_3/BiasAdd_2/ReadVariableOp2R
'backbone/conv2d_3/Conv2D/ReadVariableOp'backbone/conv2d_3/Conv2D/ReadVariableOp2V
)backbone/conv2d_3/Conv2D_1/ReadVariableOp)backbone/conv2d_3/Conv2D_1/ReadVariableOp2V
)backbone/conv2d_3/Conv2D_2/ReadVariableOp)backbone/conv2d_3/Conv2D_2/ReadVariableOp2N
%backbone/dense/BiasAdd/ReadVariableOp%backbone/dense/BiasAdd/ReadVariableOp2R
'backbone/dense/BiasAdd_1/ReadVariableOp'backbone/dense/BiasAdd_1/ReadVariableOp2R
'backbone/dense/BiasAdd_2/ReadVariableOp'backbone/dense/BiasAdd_2/ReadVariableOp2L
$backbone/dense/MatMul/ReadVariableOp$backbone/dense/MatMul/ReadVariableOp2P
&backbone/dense/MatMul_1/ReadVariableOp&backbone/dense/MatMul_1/ReadVariableOp2P
&backbone/dense/MatMul_2/ReadVariableOp&backbone/dense/MatMul_2/ReadVariableOp:Z V
0
_output_shapes
:����������~
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:����������~
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:����������~
"
_user_specified_name
inputs/2
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1187089

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
e
I__inference_activation_3_layer_call_and_return_conditional_losses_1184288

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1187519

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� @:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1187662

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1184512

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� @:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:��������� @�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1187298

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@?@:@:@:@:@:*
epsilon%o�:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������@?@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@?@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������@?@
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_layer_call_fn_1187058

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������~@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1184105x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������~@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������~@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������~@
 
_user_specified_nameinputs
�-
�
D__inference_triplet_layer_call_and_return_conditional_losses_1185243

inputs
inputs_1
inputs_2*
backbone_1185070:@
backbone_1185072:@
backbone_1185074:@
backbone_1185076:@
backbone_1185078:@
backbone_1185080:@*
backbone_1185082:@@
backbone_1185084:@
backbone_1185086:@
backbone_1185088:@
backbone_1185090:@
backbone_1185092:@*
backbone_1185094:@@
backbone_1185096:@
backbone_1185098:@
backbone_1185100:@
backbone_1185102:@
backbone_1185104:@*
backbone_1185106:@@
backbone_1185108:@
backbone_1185110:@
backbone_1185112:@
backbone_1185114:@
backbone_1185116:@#
backbone_1185118:	�@
backbone_1185120:@
identity�� backbone/StatefulPartitionedCall�"backbone/StatefulPartitionedCall_1�"backbone/StatefulPartitionedCall_2�
 backbone/StatefulPartitionedCallStatefulPartitionedCallinputsbackbone_1185070backbone_1185072backbone_1185074backbone_1185076backbone_1185078backbone_1185080backbone_1185082backbone_1185084backbone_1185086backbone_1185088backbone_1185090backbone_1185092backbone_1185094backbone_1185096backbone_1185098backbone_1185100backbone_1185102backbone_1185104backbone_1185106backbone_1185108backbone_1185110backbone_1185112backbone_1185114backbone_1185116backbone_1185118backbone_1185120*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_backbone_layer_call_and_return_conditional_losses_1184327�
"backbone/StatefulPartitionedCall_1StatefulPartitionedCallinputs_2backbone_1185070backbone_1185072backbone_1185074backbone_1185076backbone_1185078backbone_1185080backbone_1185082backbone_1185084backbone_1185086backbone_1185088backbone_1185090backbone_1185092backbone_1185094backbone_1185096backbone_1185098backbone_1185100backbone_1185102backbone_1185104backbone_1185106backbone_1185108backbone_1185110backbone_1185112backbone_1185114backbone_1185116backbone_1185118backbone_1185120*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_backbone_layer_call_and_return_conditional_losses_1184327�
"backbone/StatefulPartitionedCall_2StatefulPartitionedCallinputs_1backbone_1185070backbone_1185072backbone_1185074backbone_1185076backbone_1185078backbone_1185080backbone_1185082backbone_1185084backbone_1185086backbone_1185088backbone_1185090backbone_1185092backbone_1185094backbone_1185096backbone_1185098backbone_1185100backbone_1185102backbone_1185104backbone_1185106backbone_1185108backbone_1185110backbone_1185112backbone_1185114backbone_1185116backbone_1185118backbone_1185120*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_backbone_layer_call_and_return_conditional_losses_1184327�
dot/PartitionedCallPartitionedCall)backbone/StatefulPartitionedCall:output:0+backbone/StatefulPartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dot_layer_call_and_return_conditional_losses_1185203�
dot_1/PartitionedCallPartitionedCall)backbone/StatefulPartitionedCall:output:0+backbone/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dot_1_layer_call_and_return_conditional_losses_1185231�
concatenate/PartitionedCallPartitionedCalldot/PartitionedCall:output:0dot_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1185240s
IdentityIdentity$concatenate/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^backbone/StatefulPartitionedCall#^backbone/StatefulPartitionedCall_1#^backbone/StatefulPartitionedCall_2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������~:����������~:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 backbone/StatefulPartitionedCall backbone/StatefulPartitionedCall2H
"backbone/StatefulPartitionedCall_1"backbone/StatefulPartitionedCall_12H
"backbone/StatefulPartitionedCall_2"backbone/StatefulPartitionedCall_2:X T
0
_output_shapes
:����������~
 
_user_specified_nameinputs:XT
0
_output_shapes
:����������~
 
_user_specified_nameinputs:XT
0
_output_shapes
:����������~
 
_user_specified_nameinputs
�
e
I__inference_activation_2_layer_call_and_return_conditional_losses_1184232

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:��������� @b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:��������� @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� @:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1187125

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:����������~@:@:@:@:@:*
epsilon%o�:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:����������~@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������~@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:����������~@
 
_user_specified_nameinputs
�
f
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1187173

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������@?@*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������@?@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������~@:X T
0
_output_shapes
:����������~@
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_4_layer_call_fn_1187702

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1184300h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
)__inference_triplet_layer_call_fn_1186029
inputs_0
inputs_1
inputs_2!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	�@

unknown_24:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_triplet_layer_call_and_return_conditional_losses_1185498o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������~:����������~:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:����������~
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:����������~
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:����������~
"
_user_specified_name
inputs/2
�
S
'__inference_dot_1_layer_call_fn_1186961
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dot_1_layer_call_and_return_conditional_losses_1185231`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs/1
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1183771

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
)__inference_triplet_layer_call_fn_1185298
anchor_input
positive_input
negative_input!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	�@

unknown_24:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallanchor_inputpositive_inputnegative_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_triplet_layer_call_and_return_conditional_losses_1185243o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������~:����������~:����������~: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:����������~
&
_user_specified_nameanchor_input:`\
0
_output_shapes
:����������~
(
_user_specified_namepositive_input:`\
0
_output_shapes
:����������~
(
_user_specified_namenegative_input
�
h
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1187692

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
'__inference_dense_layer_call_fn_1187732

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1184320o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_3_layer_call_fn_1187528

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1184250w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_3_layer_call_fn_1187677

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1184050�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
e
I__inference_activation_2_layer_call_and_return_conditional_losses_1187499

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:��������� @b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:��������� @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� @:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_3_layer_call_fn_1187590

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1184447w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
f
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1183822

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
N
anchor_input>
serving_default_anchor_input:0����������~
R
negative_input@
 serving_default_negative_input:0����������~
R
positive_input@
 serving_default_positive_input:0����������~?
concatenate0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer-5
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
layer-8
layer_with_weights-4
layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer-16
layer-17
 layer-18
!layer_with_weights-8
!layer-19
"	variables
#trainable_variables
$regularization_losses
%	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_network
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
2iter

3beta_1

4beta_2
	5decay
6learning_rate7m�8m�9m�:m�=m�>m�?m�@m�Cm�Dm�Em�Fm�Im�Jm�Km�Lm�Om�Pm�7v�8v�9v�:v�=v�>v�?v�@v�Cv�Dv�Ev�Fv�Iv�Jv�Kv�Lv�Ov�Pv�"
	optimizer
�
70
81
92
:3
;4
<5
=6
>7
?8
@9
A10
B11
C12
D13
E14
F15
G16
H17
I18
J19
K20
L21
M22
N23
O24
P25"
trackable_list_wrapper
�
70
81
92
:3
=4
>5
?6
@7
C8
D9
E10
F11
I12
J13
K14
L15
O16
P17"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
		variables

trainable_variables
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
"
_tf_keras_input_layer
�

7kernel
8bias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Zaxis
	9gamma
:beta
;moving_mean
<moving_variance
[	variables
\trainable_variables
]regularization_losses
^	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

=kernel
>bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
kaxis
	?gamma
@beta
Amoving_mean
Bmoving_variance
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Ckernel
Dbias
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
|axis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Ikernel
Jbias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	�axis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Okernel
Pbias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
70
81
92
:3
;4
<5
=6
>7
?8
@9
A10
B11
C12
D13
E14
F15
G16
H17
I18
J19
K20
L21
M22
N23
O24
P25"
trackable_list_wrapper
�
70
81
92
:3
=4
>5
?6
@7
C8
D9
E10
F11
I12
J13
K14
L15
O16
P17"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
&	variables
'trainable_variables
(regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
':%@2conv2d/kernel
:@2conv2d/bias
':%@2batch_normalization/gamma
&:$@2batch_normalization/beta
/:-@ (2batch_normalization/moving_mean
3:1@ (2#batch_normalization/moving_variance
):'@@2conv2d_1/kernel
:@2conv2d_1/bias
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
):'@@2conv2d_2/kernel
:@2conv2d_2/bias
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
):'@@2conv2d_3/kernel
:@2conv2d_3/bias
):'@2batch_normalization_3/gamma
(:&@2batch_normalization_3/beta
1:/@ (2!batch_normalization_3/moving_mean
5:3@ (2%batch_normalization_3/moving_variance
:	�@2dense/kernel
:@2
dense/bias
X
;0
<1
A2
B3
G4
H5
M6
N7"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
90
:1
;2
<3"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
?0
@1
A2
B3"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
E0
F1
G2
H3"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
K0
L1
M2
N3"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
X
;0
<1
A2
B3
G4
H5
M6
N7"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
 18
!19"
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
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
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
.
;0
<1"
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
.
A0
B1"
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
.
G0
H1"
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
.
M0
N1"
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
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
,:*@2Adam/conv2d/kernel/m
:@2Adam/conv2d/bias/m
,:*@2 Adam/batch_normalization/gamma/m
+:)@2Adam/batch_normalization/beta/m
.:,@@2Adam/conv2d_1/kernel/m
 :@2Adam/conv2d_1/bias/m
.:,@2"Adam/batch_normalization_1/gamma/m
-:+@2!Adam/batch_normalization_1/beta/m
.:,@@2Adam/conv2d_2/kernel/m
 :@2Adam/conv2d_2/bias/m
.:,@2"Adam/batch_normalization_2/gamma/m
-:+@2!Adam/batch_normalization_2/beta/m
.:,@@2Adam/conv2d_3/kernel/m
 :@2Adam/conv2d_3/bias/m
.:,@2"Adam/batch_normalization_3/gamma/m
-:+@2!Adam/batch_normalization_3/beta/m
$:"	�@2Adam/dense/kernel/m
:@2Adam/dense/bias/m
,:*@2Adam/conv2d/kernel/v
:@2Adam/conv2d/bias/v
,:*@2 Adam/batch_normalization/gamma/v
+:)@2Adam/batch_normalization/beta/v
.:,@@2Adam/conv2d_1/kernel/v
 :@2Adam/conv2d_1/bias/v
.:,@2"Adam/batch_normalization_1/gamma/v
-:+@2!Adam/batch_normalization_1/beta/v
.:,@@2Adam/conv2d_2/kernel/v
 :@2Adam/conv2d_2/bias/v
.:,@2"Adam/batch_normalization_2/gamma/v
-:+@2!Adam/batch_normalization_2/beta/v
.:,@@2Adam/conv2d_3/kernel/v
 :@2Adam/conv2d_3/bias/v
.:,@2"Adam/batch_normalization_3/gamma/v
-:+@2!Adam/batch_normalization_3/beta/v
$:"	�@2Adam/dense/kernel/v
:@2Adam/dense/bias/v
�2�
)__inference_triplet_layer_call_fn_1185298
)__inference_triplet_layer_call_fn_1185970
)__inference_triplet_layer_call_fn_1186029
)__inference_triplet_layer_call_fn_1185612�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_triplet_layer_call_and_return_conditional_losses_1186318
D__inference_triplet_layer_call_and_return_conditional_losses_1186607
D__inference_triplet_layer_call_and_return_conditional_losses_1185728
D__inference_triplet_layer_call_and_return_conditional_losses_1185844�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
"__inference__wrapped_model_1183749anchor_inputpositive_inputnegative_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_backbone_layer_call_fn_1184382
*__inference_backbone_layer_call_fn_1186664
*__inference_backbone_layer_call_fn_1186721
*__inference_backbone_layer_call_fn_1184909�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_backbone_layer_call_and_return_conditional_losses_1186822
E__inference_backbone_layer_call_and_return_conditional_losses_1186923
E__inference_backbone_layer_call_and_return_conditional_losses_1184984
E__inference_backbone_layer_call_and_return_conditional_losses_1185059�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
%__inference_dot_layer_call_fn_1186929�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_dot_layer_call_and_return_conditional_losses_1186955�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dot_1_layer_call_fn_1186961�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dot_1_layer_call_and_return_conditional_losses_1186987�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_concatenate_layer_call_fn_1186993�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_concatenate_layer_call_and_return_conditional_losses_1187000�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_1185911anchor_inputnegative_inputpositive_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_conv2d_layer_call_fn_1187009�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_conv2d_layer_call_and_return_conditional_losses_1187019�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_batch_normalization_layer_call_fn_1187032
5__inference_batch_normalization_layer_call_fn_1187045
5__inference_batch_normalization_layer_call_fn_1187058
5__inference_batch_normalization_layer_call_fn_1187071�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1187089
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1187107
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1187125
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1187143�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_activation_layer_call_fn_1187148�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_activation_layer_call_and_return_conditional_losses_1187153�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_max_pooling2d_layer_call_fn_1187158
/__inference_max_pooling2d_layer_call_fn_1187163�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1187168
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1187173�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_1_layer_call_fn_1187182�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1187192�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
7__inference_batch_normalization_1_layer_call_fn_1187205
7__inference_batch_normalization_1_layer_call_fn_1187218
7__inference_batch_normalization_1_layer_call_fn_1187231
7__inference_batch_normalization_1_layer_call_fn_1187244�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1187262
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1187280
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1187298
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1187316�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
.__inference_activation_1_layer_call_fn_1187321�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_activation_1_layer_call_and_return_conditional_losses_1187326�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_max_pooling2d_1_layer_call_fn_1187331
1__inference_max_pooling2d_1_layer_call_fn_1187336�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1187341
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1187346�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_2_layer_call_fn_1187355�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1187365�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
7__inference_batch_normalization_2_layer_call_fn_1187378
7__inference_batch_normalization_2_layer_call_fn_1187391
7__inference_batch_normalization_2_layer_call_fn_1187404
7__inference_batch_normalization_2_layer_call_fn_1187417�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1187435
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1187453
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1187471
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1187489�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
.__inference_activation_2_layer_call_fn_1187494�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_activation_2_layer_call_and_return_conditional_losses_1187499�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_max_pooling2d_2_layer_call_fn_1187504
1__inference_max_pooling2d_2_layer_call_fn_1187509�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1187514
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1187519�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_3_layer_call_fn_1187528�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1187538�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
7__inference_batch_normalization_3_layer_call_fn_1187551
7__inference_batch_normalization_3_layer_call_fn_1187564
7__inference_batch_normalization_3_layer_call_fn_1187577
7__inference_batch_normalization_3_layer_call_fn_1187590�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1187608
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1187626
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1187644
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1187662�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
.__inference_activation_3_layer_call_fn_1187667�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_activation_3_layer_call_and_return_conditional_losses_1187672�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_max_pooling2d_3_layer_call_fn_1187677
1__inference_max_pooling2d_3_layer_call_fn_1187682�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1187687
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1187692�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_max_pooling2d_4_layer_call_fn_1187697
1__inference_max_pooling2d_4_layer_call_fn_1187702�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1187707
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1187712�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_flatten_layer_call_fn_1187717�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_flatten_layer_call_and_return_conditional_losses_1187723�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_layer_call_fn_1187732�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_layer_call_and_return_conditional_losses_1187742�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
"__inference__wrapped_model_1183749�789:;<=>?@ABCDEFGHIJKLMNOP���
���
���
/�,
anchor_input����������~
1�.
positive_input����������~
1�.
negative_input����������~
� "9�6
4
concatenate%�"
concatenate����������
I__inference_activation_1_layer_call_and_return_conditional_losses_1187326h7�4
-�*
(�%
inputs���������@?@
� "-�*
#� 
0���������@?@
� �
.__inference_activation_1_layer_call_fn_1187321[7�4
-�*
(�%
inputs���������@?@
� " ����������@?@�
I__inference_activation_2_layer_call_and_return_conditional_losses_1187499h7�4
-�*
(�%
inputs��������� @
� "-�*
#� 
0��������� @
� �
.__inference_activation_2_layer_call_fn_1187494[7�4
-�*
(�%
inputs��������� @
� " ���������� @�
I__inference_activation_3_layer_call_and_return_conditional_losses_1187672h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
.__inference_activation_3_layer_call_fn_1187667[7�4
-�*
(�%
inputs���������@
� " ����������@�
G__inference_activation_layer_call_and_return_conditional_losses_1187153j8�5
.�+
)�&
inputs����������~@
� ".�+
$�!
0����������~@
� �
,__inference_activation_layer_call_fn_1187148]8�5
.�+
)�&
inputs����������~@
� "!�����������~@�
E__inference_backbone_layer_call_and_return_conditional_losses_1184984�789:;<=>?@ABCDEFGHIJKLMNOPA�>
7�4
*�'
input_1����������~
p 

 
� "%�"
�
0���������@
� �
E__inference_backbone_layer_call_and_return_conditional_losses_1185059�789:;<=>?@ABCDEFGHIJKLMNOPA�>
7�4
*�'
input_1����������~
p

 
� "%�"
�
0���������@
� �
E__inference_backbone_layer_call_and_return_conditional_losses_1186822�789:;<=>?@ABCDEFGHIJKLMNOP@�=
6�3
)�&
inputs����������~
p 

 
� "%�"
�
0���������@
� �
E__inference_backbone_layer_call_and_return_conditional_losses_1186923�789:;<=>?@ABCDEFGHIJKLMNOP@�=
6�3
)�&
inputs����������~
p

 
� "%�"
�
0���������@
� �
*__inference_backbone_layer_call_fn_1184382y789:;<=>?@ABCDEFGHIJKLMNOPA�>
7�4
*�'
input_1����������~
p 

 
� "����������@�
*__inference_backbone_layer_call_fn_1184909y789:;<=>?@ABCDEFGHIJKLMNOPA�>
7�4
*�'
input_1����������~
p

 
� "����������@�
*__inference_backbone_layer_call_fn_1186664x789:;<=>?@ABCDEFGHIJKLMNOP@�=
6�3
)�&
inputs����������~
p 

 
� "����������@�
*__inference_backbone_layer_call_fn_1186721x789:;<=>?@ABCDEFGHIJKLMNOP@�=
6�3
)�&
inputs����������~
p

 
� "����������@�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1187262�?@ABM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1187280�?@ABM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1187298r?@AB;�8
1�.
(�%
inputs���������@?@
p 
� "-�*
#� 
0���������@?@
� �
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1187316r?@AB;�8
1�.
(�%
inputs���������@?@
p
� "-�*
#� 
0���������@?@
� �
7__inference_batch_normalization_1_layer_call_fn_1187205�?@ABM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
7__inference_batch_normalization_1_layer_call_fn_1187218�?@ABM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_1_layer_call_fn_1187231e?@AB;�8
1�.
(�%
inputs���������@?@
p 
� " ����������@?@�
7__inference_batch_normalization_1_layer_call_fn_1187244e?@AB;�8
1�.
(�%
inputs���������@?@
p
� " ����������@?@�
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1187435�EFGHM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1187453�EFGHM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1187471rEFGH;�8
1�.
(�%
inputs��������� @
p 
� "-�*
#� 
0��������� @
� �
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1187489rEFGH;�8
1�.
(�%
inputs��������� @
p
� "-�*
#� 
0��������� @
� �
7__inference_batch_normalization_2_layer_call_fn_1187378�EFGHM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
7__inference_batch_normalization_2_layer_call_fn_1187391�EFGHM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_2_layer_call_fn_1187404eEFGH;�8
1�.
(�%
inputs��������� @
p 
� " ���������� @�
7__inference_batch_normalization_2_layer_call_fn_1187417eEFGH;�8
1�.
(�%
inputs��������� @
p
� " ���������� @�
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1187608�KLMNM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1187626�KLMNM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1187644rKLMN;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1187662rKLMN;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
7__inference_batch_normalization_3_layer_call_fn_1187551�KLMNM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
7__inference_batch_normalization_3_layer_call_fn_1187564�KLMNM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_3_layer_call_fn_1187577eKLMN;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
7__inference_batch_normalization_3_layer_call_fn_1187590eKLMN;�8
1�.
(�%
inputs���������@
p
� " ����������@�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1187089�9:;<M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1187107�9:;<M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1187125t9:;<<�9
2�/
)�&
inputs����������~@
p 
� ".�+
$�!
0����������~@
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1187143t9:;<<�9
2�/
)�&
inputs����������~@
p
� ".�+
$�!
0����������~@
� �
5__inference_batch_normalization_layer_call_fn_1187032�9:;<M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
5__inference_batch_normalization_layer_call_fn_1187045�9:;<M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
5__inference_batch_normalization_layer_call_fn_1187058g9:;<<�9
2�/
)�&
inputs����������~@
p 
� "!�����������~@�
5__inference_batch_normalization_layer_call_fn_1187071g9:;<<�9
2�/
)�&
inputs����������~@
p
� "!�����������~@�
H__inference_concatenate_layer_call_and_return_conditional_losses_1187000�Z�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "%�"
�
0���������
� �
-__inference_concatenate_layer_call_fn_1186993vZ�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "�����������
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1187192l=>7�4
-�*
(�%
inputs���������@?@
� "-�*
#� 
0���������@?@
� �
*__inference_conv2d_1_layer_call_fn_1187182_=>7�4
-�*
(�%
inputs���������@?@
� " ����������@?@�
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1187365lCD7�4
-�*
(�%
inputs��������� @
� "-�*
#� 
0��������� @
� �
*__inference_conv2d_2_layer_call_fn_1187355_CD7�4
-�*
(�%
inputs��������� @
� " ���������� @�
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1187538lIJ7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
*__inference_conv2d_3_layer_call_fn_1187528_IJ7�4
-�*
(�%
inputs���������@
� " ����������@�
C__inference_conv2d_layer_call_and_return_conditional_losses_1187019n788�5
.�+
)�&
inputs����������~
� ".�+
$�!
0����������~@
� �
(__inference_conv2d_layer_call_fn_1187009a788�5
.�+
)�&
inputs����������~
� "!�����������~@�
B__inference_dense_layer_call_and_return_conditional_losses_1187742]OP0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� {
'__inference_dense_layer_call_fn_1187732POP0�-
&�#
!�
inputs����������
� "����������@�
B__inference_dot_1_layer_call_and_return_conditional_losses_1186987�Z�W
P�M
K�H
"�
inputs/0���������@
"�
inputs/1���������@
� "%�"
�
0���������
� �
'__inference_dot_1_layer_call_fn_1186961vZ�W
P�M
K�H
"�
inputs/0���������@
"�
inputs/1���������@
� "�����������
@__inference_dot_layer_call_and_return_conditional_losses_1186955�Z�W
P�M
K�H
"�
inputs/0���������@
"�
inputs/1���������@
� "%�"
�
0���������
� �
%__inference_dot_layer_call_fn_1186929vZ�W
P�M
K�H
"�
inputs/0���������@
"�
inputs/1���������@
� "�����������
D__inference_flatten_layer_call_and_return_conditional_losses_1187723a7�4
-�*
(�%
inputs���������@
� "&�#
�
0����������
� �
)__inference_flatten_layer_call_fn_1187717T7�4
-�*
(�%
inputs���������@
� "������������
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1187341�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1187346h7�4
-�*
(�%
inputs���������@?@
� "-�*
#� 
0��������� @
� �
1__inference_max_pooling2d_1_layer_call_fn_1187331�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
1__inference_max_pooling2d_1_layer_call_fn_1187336[7�4
-�*
(�%
inputs���������@?@
� " ���������� @�
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1187514�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1187519h7�4
-�*
(�%
inputs��������� @
� "-�*
#� 
0���������@
� �
1__inference_max_pooling2d_2_layer_call_fn_1187504�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
1__inference_max_pooling2d_2_layer_call_fn_1187509[7�4
-�*
(�%
inputs��������� @
� " ����������@�
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1187687�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
L__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1187692h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
1__inference_max_pooling2d_3_layer_call_fn_1187677�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
1__inference_max_pooling2d_3_layer_call_fn_1187682[7�4
-�*
(�%
inputs���������@
� " ����������@�
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1187707�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_1187712h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
1__inference_max_pooling2d_4_layer_call_fn_1187697�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
1__inference_max_pooling2d_4_layer_call_fn_1187702[7�4
-�*
(�%
inputs���������@
� " ����������@�
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1187168�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1187173i8�5
.�+
)�&
inputs����������~@
� "-�*
#� 
0���������@?@
� �
/__inference_max_pooling2d_layer_call_fn_1187158�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
/__inference_max_pooling2d_layer_call_fn_1187163\8�5
.�+
)�&
inputs����������~@
� " ����������@?@�
%__inference_signature_wrapper_1185911�789:;<=>?@ABCDEFGHIJKLMNOP���
� 
���
?
anchor_input/�,
anchor_input����������~
C
negative_input1�.
negative_input����������~
C
positive_input1�.
positive_input����������~"9�6
4
concatenate%�"
concatenate����������
D__inference_triplet_layer_call_and_return_conditional_losses_1185728�789:;<=>?@ABCDEFGHIJKLMNOP���
���
���
/�,
anchor_input����������~
1�.
positive_input����������~
1�.
negative_input����������~
p 

 
� "%�"
�
0���������
� �
D__inference_triplet_layer_call_and_return_conditional_losses_1185844�789:;<=>?@ABCDEFGHIJKLMNOP���
���
���
/�,
anchor_input����������~
1�.
positive_input����������~
1�.
negative_input����������~
p

 
� "%�"
�
0���������
� �
D__inference_triplet_layer_call_and_return_conditional_losses_1186318�789:;<=>?@ABCDEFGHIJKLMNOP���
���
���
+�(
inputs/0����������~
+�(
inputs/1����������~
+�(
inputs/2����������~
p 

 
� "%�"
�
0���������
� �
D__inference_triplet_layer_call_and_return_conditional_losses_1186607�789:;<=>?@ABCDEFGHIJKLMNOP���
���
���
+�(
inputs/0����������~
+�(
inputs/1����������~
+�(
inputs/2����������~
p

 
� "%�"
�
0���������
� �
)__inference_triplet_layer_call_fn_1185298�789:;<=>?@ABCDEFGHIJKLMNOP���
���
���
/�,
anchor_input����������~
1�.
positive_input����������~
1�.
negative_input����������~
p 

 
� "�����������
)__inference_triplet_layer_call_fn_1185612�789:;<=>?@ABCDEFGHIJKLMNOP���
���
���
/�,
anchor_input����������~
1�.
positive_input����������~
1�.
negative_input����������~
p

 
� "�����������
)__inference_triplet_layer_call_fn_1185970�789:;<=>?@ABCDEFGHIJKLMNOP���
���
���
+�(
inputs/0����������~
+�(
inputs/1����������~
+�(
inputs/2����������~
p 

 
� "�����������
)__inference_triplet_layer_call_fn_1186029�789:;<=>?@ABCDEFGHIJKLMNOP���
���
���
+�(
inputs/0����������~
+�(
inputs/1����������~
+�(
inputs/2����������~
p

 
� "����������