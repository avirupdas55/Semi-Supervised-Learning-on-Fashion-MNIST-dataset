??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
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
.
Identity

input"T
output"T"	
Ttype
?
	MLCConv2D

input"T
filter"T

unique_key"T*num_args
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)
"
	transposebool( "
num_argsint(
?
	MLCMatMul
a"T
b"T

unique_key"T*num_args
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2"
num_argsint ("

input_rankint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*	2.4.0-rc02&tf_macos-v0.1-alpha2-AS-67-gf3595294ab8??	
?
conv2d_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_51/kernel
}
$conv2d_51/kernel/Read/ReadVariableOpReadVariableOpconv2d_51/kernel*&
_output_shapes
: *
dtype0
t
conv2d_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_51/bias
m
"conv2d_51/bias/Read/ReadVariableOpReadVariableOpconv2d_51/bias*
_output_shapes
: *
dtype0
?
conv2d_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_52/kernel
}
$conv2d_52/kernel/Read/ReadVariableOpReadVariableOpconv2d_52/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_52/bias
m
"conv2d_52/bias/Read/ReadVariableOpReadVariableOpconv2d_52/bias*
_output_shapes
:@*
dtype0
?
conv2d_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*!
shared_nameconv2d_53/kernel
~
$conv2d_53/kernel/Read/ReadVariableOpReadVariableOpconv2d_53/kernel*'
_output_shapes
:@?*
dtype0
u
conv2d_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_53/bias
n
"conv2d_53/bias/Read/ReadVariableOpReadVariableOpconv2d_53/bias*
_output_shapes	
:?*
dtype0
?
conv1d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameconv1d_17/kernel
z
$conv1d_17/kernel/Read/ReadVariableOpReadVariableOpconv1d_17/kernel*#
_output_shapes
:?*
dtype0
t
conv1d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_17/bias
m
"conv1d_17/bias/Read/ReadVariableOpReadVariableOpconv1d_17/bias*
_output_shapes
:*
dtype0
{
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@* 
shared_namedense_34/kernel
t
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel*
_output_shapes
:	?@*
dtype0
r
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_34/bias
k
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
_output_shapes
:@*
dtype0
z
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
* 
shared_namedense_35/kernel
s
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes

:@
*
dtype0
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
_output_shapes
:
*
dtype0
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv2d_51/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_51/kernel/m
?
+Adam/conv2d_51/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_51/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_51/bias/m
{
)Adam/conv2d_51/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_52/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_52/kernel/m
?
+Adam/conv2d_52/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_52/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_52/bias/m
{
)Adam/conv2d_52/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_53/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_53/kernel/m
?
+Adam/conv2d_53/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_53/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_53/bias/m
|
)Adam/conv2d_53/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv1d_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/conv1d_17/kernel/m
?
+Adam/conv1d_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_17/kernel/m*#
_output_shapes
:?*
dtype0
?
Adam/conv1d_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1d_17/bias/m
{
)Adam/conv1d_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_17/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/dense_34/kernel/m
?
*Adam/dense_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/m*
_output_shapes
:	?@*
dtype0
?
Adam/dense_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_34/bias/m
y
(Adam/dense_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*'
shared_nameAdam/dense_35/kernel/m
?
*Adam/dense_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/m*
_output_shapes

:@
*
dtype0
?
Adam/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_35/bias/m
y
(Adam/dense_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv2d_51/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_51/kernel/v
?
+Adam/conv2d_51/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_51/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_51/bias/v
{
)Adam/conv2d_51/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_52/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_52/kernel/v
?
+Adam/conv2d_52/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_52/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_52/bias/v
{
)Adam/conv2d_52/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_53/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_53/kernel/v
?
+Adam/conv2d_53/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_53/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_53/bias/v
|
)Adam/conv2d_53/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv1d_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/conv1d_17/kernel/v
?
+Adam/conv1d_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_17/kernel/v*#
_output_shapes
:?*
dtype0
?
Adam/conv1d_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1d_17/bias/v
{
)Adam/conv1d_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_17/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/dense_34/kernel/v
?
*Adam/dense_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/v*
_output_shapes
:	?@*
dtype0
?
Adam/dense_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_34/bias/v
y
(Adam/dense_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*'
shared_nameAdam/dense_35/kernel/v
?
*Adam/dense_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/v*
_output_shapes

:@
*
dtype0
?
Adam/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_35/bias/v
y
(Adam/dense_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?B
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?A
value?AB?A B?A
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
R
&	variables
'trainable_variables
(regularization_losses
)	keras_api
h

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
h

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
?
6iter

7beta_1

8beta_2
	9decay
:learning_ratemnmompmqmrms mt!mu*mv+mw0mx1myvzv{v|v}v~v v?!v?*v?+v?0v?1v?
V
0
1
2
3
4
5
 6
!7
*8
+9
010
111
V
0
1
2
3
4
5
 6
!7
*8
+9
010
111
 
?
;non_trainable_variables
<layer_metrics
=metrics

>layers
		variables
?layer_regularization_losses

trainable_variables
regularization_losses
 
\Z
VARIABLE_VALUEconv2d_51/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_51/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
@non_trainable_variables
Alayer_metrics
Bmetrics

Clayers
	variables
Dlayer_regularization_losses
trainable_variables
regularization_losses
\Z
VARIABLE_VALUEconv2d_52/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_52/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Enon_trainable_variables
Flayer_metrics
Gmetrics

Hlayers
	variables
Ilayer_regularization_losses
trainable_variables
regularization_losses
\Z
VARIABLE_VALUEconv2d_53/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_53/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Jnon_trainable_variables
Klayer_metrics
Lmetrics

Mlayers
	variables
Nlayer_regularization_losses
trainable_variables
regularization_losses
\Z
VARIABLE_VALUEconv1d_17/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_17/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
?
Onon_trainable_variables
Player_metrics
Qmetrics

Rlayers
"	variables
Slayer_regularization_losses
#trainable_variables
$regularization_losses
 
 
 
?
Tnon_trainable_variables
Ulayer_metrics
Vmetrics

Wlayers
&	variables
Xlayer_regularization_losses
'trainable_variables
(regularization_losses
[Y
VARIABLE_VALUEdense_34/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_34/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
?
Ynon_trainable_variables
Zlayer_metrics
[metrics

\layers
,	variables
]layer_regularization_losses
-trainable_variables
.regularization_losses
[Y
VARIABLE_VALUEdense_35/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_35/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
 
?
^non_trainable_variables
_layer_metrics
`metrics

alayers
2	variables
blayer_regularization_losses
3trainable_variables
4regularization_losses
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
 
 

c0
d1
1
0
1
2
3
4
5
6
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
 
 
 
 
 
 
 
4
	etotal
	fcount
g	variables
h	keras_api
D
	itotal
	jcount
k
_fn_kwargs
l	variables
m	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

e0
f1

g	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

i0
j1

l	variables
}
VARIABLE_VALUEAdam/conv2d_51/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_51/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_52/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_52/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_53/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_53/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_17/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_17/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_34/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_34/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_35/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_35/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_51/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_51/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_52/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_52/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_53/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_53/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_17/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_17/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_34/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_34/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_35/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_35/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_18Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_18conv2d_51/kernelconv2d_51/biasconv2d_52/kernelconv2d_52/biasconv2d_53/kernelconv2d_53/biasconv1d_17/kernelconv1d_17/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_425284
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_51/kernel/Read/ReadVariableOp"conv2d_51/bias/Read/ReadVariableOp$conv2d_52/kernel/Read/ReadVariableOp"conv2d_52/bias/Read/ReadVariableOp$conv2d_53/kernel/Read/ReadVariableOp"conv2d_53/bias/Read/ReadVariableOp$conv1d_17/kernel/Read/ReadVariableOp"conv1d_17/bias/Read/ReadVariableOp#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOp#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_51/kernel/m/Read/ReadVariableOp)Adam/conv2d_51/bias/m/Read/ReadVariableOp+Adam/conv2d_52/kernel/m/Read/ReadVariableOp)Adam/conv2d_52/bias/m/Read/ReadVariableOp+Adam/conv2d_53/kernel/m/Read/ReadVariableOp)Adam/conv2d_53/bias/m/Read/ReadVariableOp+Adam/conv1d_17/kernel/m/Read/ReadVariableOp)Adam/conv1d_17/bias/m/Read/ReadVariableOp*Adam/dense_34/kernel/m/Read/ReadVariableOp(Adam/dense_34/bias/m/Read/ReadVariableOp*Adam/dense_35/kernel/m/Read/ReadVariableOp(Adam/dense_35/bias/m/Read/ReadVariableOp+Adam/conv2d_51/kernel/v/Read/ReadVariableOp)Adam/conv2d_51/bias/v/Read/ReadVariableOp+Adam/conv2d_52/kernel/v/Read/ReadVariableOp)Adam/conv2d_52/bias/v/Read/ReadVariableOp+Adam/conv2d_53/kernel/v/Read/ReadVariableOp)Adam/conv2d_53/bias/v/Read/ReadVariableOp+Adam/conv1d_17/kernel/v/Read/ReadVariableOp)Adam/conv1d_17/bias/v/Read/ReadVariableOp*Adam/dense_34/kernel/v/Read/ReadVariableOp(Adam/dense_34/bias/v/Read/ReadVariableOp*Adam/dense_35/kernel/v/Read/ReadVariableOp(Adam/dense_35/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_425808
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_51/kernelconv2d_51/biasconv2d_52/kernelconv2d_52/biasconv2d_53/kernelconv2d_53/biasconv1d_17/kernelconv1d_17/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_51/kernel/mAdam/conv2d_51/bias/mAdam/conv2d_52/kernel/mAdam/conv2d_52/bias/mAdam/conv2d_53/kernel/mAdam/conv2d_53/bias/mAdam/conv1d_17/kernel/mAdam/conv1d_17/bias/mAdam/dense_34/kernel/mAdam/dense_34/bias/mAdam/dense_35/kernel/mAdam/dense_35/bias/mAdam/conv2d_51/kernel/vAdam/conv2d_51/bias/vAdam/conv2d_52/kernel/vAdam/conv2d_52/bias/vAdam/conv2d_53/kernel/vAdam/conv2d_53/bias/vAdam/conv1d_17/kernel/vAdam/conv1d_17/bias/vAdam/dense_34/kernel/vAdam/dense_34/bias/vAdam/dense_35/kernel/vAdam/dense_35/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_425953??
?

?
E__inference_conv2d_53_layer_call_and_return_conditional_losses_424942

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
num_args *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
D__inference_dense_34_layer_call_and_return_conditional_losses_425621

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
.__inference_sequential_17_layer_call_fn_425492

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_4252182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
~
)__inference_dense_34_layer_call_fn_425630

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_4250372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_425218

inputs
conv2d_51_425186
conv2d_51_425188
conv2d_52_425191
conv2d_52_425193
conv2d_53_425196
conv2d_53_425198
conv1d_17_425201
conv1d_17_425203
dense_34_425207
dense_34_425209
dense_35_425212
dense_35_425214
identity??!conv1d_17/StatefulPartitionedCall?!conv2d_51/StatefulPartitionedCall?!conv2d_52/StatefulPartitionedCall?!conv2d_53/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_51_425186conv2d_51_425188*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_4248882#
!conv2d_51/StatefulPartitionedCall?
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0conv2d_52_425191conv2d_52_425193*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_4249152#
!conv2d_52/StatefulPartitionedCall?
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0conv2d_53_425196conv2d_53_425198*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_53_layer_call_and_return_conditional_losses_4249422#
!conv2d_53/StatefulPartitionedCall?
!conv1d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0conv1d_17_425201conv1d_17_425203*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_17_layer_call_and_return_conditional_losses_4249962#
!conv1d_17/StatefulPartitionedCall?
flatten_17/PartitionedCallPartitionedCall*conv1d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_4250182
flatten_17/PartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_34_425207dense_34_425209*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_4250372"
 dense_34/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_425212dense_35_425214*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_4250642"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0"^conv1d_17/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::2F
!conv1d_17/StatefulPartitionedCall!conv1d_17/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_425116
input_18
conv2d_51_425084
conv2d_51_425086
conv2d_52_425089
conv2d_52_425091
conv2d_53_425094
conv2d_53_425096
conv1d_17_425099
conv1d_17_425101
dense_34_425105
dense_34_425107
dense_35_425110
dense_35_425112
identity??!conv1d_17/StatefulPartitionedCall?!conv2d_51/StatefulPartitionedCall?!conv2d_52/StatefulPartitionedCall?!conv2d_53/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCallinput_18conv2d_51_425084conv2d_51_425086*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_4248882#
!conv2d_51/StatefulPartitionedCall?
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0conv2d_52_425089conv2d_52_425091*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_4249152#
!conv2d_52/StatefulPartitionedCall?
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0conv2d_53_425094conv2d_53_425096*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_53_layer_call_and_return_conditional_losses_4249422#
!conv2d_53/StatefulPartitionedCall?
!conv1d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0conv1d_17_425099conv1d_17_425101*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_17_layer_call_and_return_conditional_losses_4249962#
!conv1d_17/StatefulPartitionedCall?
flatten_17/PartitionedCallPartitionedCall*conv1d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_4250182
flatten_17/PartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_34_425105dense_34_425107*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_4250372"
 dense_34/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_425110dense_35_425112*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_4250642"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0"^conv1d_17/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::2F
!conv1d_17/StatefulPartitionedCall!conv1d_17/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_18
?

?
E__inference_conv2d_52_layer_call_and_return_conditional_losses_424915

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
.__inference_sequential_17_layer_call_fn_425245
input_18
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_18unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_4252182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_18
?	
?
.__inference_sequential_17_layer_call_fn_425181
input_18
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_18unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_4251542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_18
?

*__inference_conv2d_51_layer_call_fn_425512

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_4248882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

*__inference_conv2d_53_layer_call_fn_425552

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_53_layer_call_and_return_conditional_losses_4249422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
b
F__inference_flatten_17_layer_call_and_return_conditional_losses_425605

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?^
?
__inference__traced_save_425808
file_prefix/
+savev2_conv2d_51_kernel_read_readvariableop-
)savev2_conv2d_51_bias_read_readvariableop/
+savev2_conv2d_52_kernel_read_readvariableop-
)savev2_conv2d_52_bias_read_readvariableop/
+savev2_conv2d_53_kernel_read_readvariableop-
)savev2_conv2d_53_bias_read_readvariableop/
+savev2_conv1d_17_kernel_read_readvariableop-
)savev2_conv1d_17_bias_read_readvariableop.
*savev2_dense_34_kernel_read_readvariableop,
(savev2_dense_34_bias_read_readvariableop.
*savev2_dense_35_kernel_read_readvariableop,
(savev2_dense_35_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_51_kernel_m_read_readvariableop4
0savev2_adam_conv2d_51_bias_m_read_readvariableop6
2savev2_adam_conv2d_52_kernel_m_read_readvariableop4
0savev2_adam_conv2d_52_bias_m_read_readvariableop6
2savev2_adam_conv2d_53_kernel_m_read_readvariableop4
0savev2_adam_conv2d_53_bias_m_read_readvariableop6
2savev2_adam_conv1d_17_kernel_m_read_readvariableop4
0savev2_adam_conv1d_17_bias_m_read_readvariableop5
1savev2_adam_dense_34_kernel_m_read_readvariableop3
/savev2_adam_dense_34_bias_m_read_readvariableop5
1savev2_adam_dense_35_kernel_m_read_readvariableop3
/savev2_adam_dense_35_bias_m_read_readvariableop6
2savev2_adam_conv2d_51_kernel_v_read_readvariableop4
0savev2_adam_conv2d_51_bias_v_read_readvariableop6
2savev2_adam_conv2d_52_kernel_v_read_readvariableop4
0savev2_adam_conv2d_52_bias_v_read_readvariableop6
2savev2_adam_conv2d_53_kernel_v_read_readvariableop4
0savev2_adam_conv2d_53_bias_v_read_readvariableop6
2savev2_adam_conv1d_17_kernel_v_read_readvariableop4
0savev2_adam_conv1d_17_bias_v_read_readvariableop5
1savev2_adam_dense_34_kernel_v_read_readvariableop3
/savev2_adam_dense_34_bias_v_read_readvariableop5
1savev2_adam_dense_35_kernel_v_read_readvariableop3
/savev2_adam_dense_35_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_51_kernel_read_readvariableop)savev2_conv2d_51_bias_read_readvariableop+savev2_conv2d_52_kernel_read_readvariableop)savev2_conv2d_52_bias_read_readvariableop+savev2_conv2d_53_kernel_read_readvariableop)savev2_conv2d_53_bias_read_readvariableop+savev2_conv1d_17_kernel_read_readvariableop)savev2_conv1d_17_bias_read_readvariableop*savev2_dense_34_kernel_read_readvariableop(savev2_dense_34_bias_read_readvariableop*savev2_dense_35_kernel_read_readvariableop(savev2_dense_35_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_51_kernel_m_read_readvariableop0savev2_adam_conv2d_51_bias_m_read_readvariableop2savev2_adam_conv2d_52_kernel_m_read_readvariableop0savev2_adam_conv2d_52_bias_m_read_readvariableop2savev2_adam_conv2d_53_kernel_m_read_readvariableop0savev2_adam_conv2d_53_bias_m_read_readvariableop2savev2_adam_conv1d_17_kernel_m_read_readvariableop0savev2_adam_conv1d_17_bias_m_read_readvariableop1savev2_adam_dense_34_kernel_m_read_readvariableop/savev2_adam_dense_34_bias_m_read_readvariableop1savev2_adam_dense_35_kernel_m_read_readvariableop/savev2_adam_dense_35_bias_m_read_readvariableop2savev2_adam_conv2d_51_kernel_v_read_readvariableop0savev2_adam_conv2d_51_bias_v_read_readvariableop2savev2_adam_conv2d_52_kernel_v_read_readvariableop0savev2_adam_conv2d_52_bias_v_read_readvariableop2savev2_adam_conv2d_53_kernel_v_read_readvariableop0savev2_adam_conv2d_53_bias_v_read_readvariableop2savev2_adam_conv1d_17_kernel_v_read_readvariableop0savev2_adam_conv1d_17_bias_v_read_readvariableop1savev2_adam_dense_34_kernel_v_read_readvariableop/savev2_adam_dense_34_bias_v_read_readvariableop1savev2_adam_dense_35_kernel_v_read_readvariableop/savev2_adam_dense_35_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : @:@:@?:?:?::	?@:@:@
:
: : : : : : : : : : : : @:@:@?:?:?::	?@:@:@
:
: : : @:@:@?:?:?::	?@:@:@
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:)%
#
_output_shapes
:?: 

_output_shapes
::%	!

_output_shapes
:	?@: 


_output_shapes
:@:$ 

_output_shapes

:@
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:)%
#
_output_shapes
:?: 

_output_shapes
::%!

_output_shapes
:	?@: 

_output_shapes
:@:$  

_output_shapes

:@
: !

_output_shapes
:
:,"(
&
_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
: @: %

_output_shapes
:@:-&)
'
_output_shapes
:@?:!'

_output_shapes	
:?:)(%
#
_output_shapes
:?: )

_output_shapes
::%*!

_output_shapes
:	?@: +

_output_shapes
:@:$, 

_output_shapes

:@
: -

_output_shapes
:
:.

_output_shapes
: 
?	
?
$__inference_signature_wrapper_425284
input_18
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_18unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_4248732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_18
?l
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_425434

inputs,
(conv2d_51_conv2d_readvariableop_resource-
)conv2d_51_biasadd_readvariableop_resource,
(conv2d_52_conv2d_readvariableop_resource-
)conv2d_52_biasadd_readvariableop_resource,
(conv2d_53_conv2d_readvariableop_resource-
)conv2d_53_biasadd_readvariableop_resource9
5conv1d_17_conv1d_expanddims_1_readvariableop_resource@
<conv1d_17_squeeze_batch_dims_biasadd_readvariableop_resource.
*dense_34_mlcmatmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource.
*dense_35_mlcmatmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity??,conv1d_17/conv1d/ExpandDims_1/ReadVariableOp?3conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOp? conv2d_51/BiasAdd/ReadVariableOp?conv2d_51/Conv2D/ReadVariableOp? conv2d_52/BiasAdd/ReadVariableOp?conv2d_52/Conv2D/ReadVariableOp? conv2d_53/BiasAdd/ReadVariableOp?conv2d_53/Conv2D/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?!dense_34/MLCMatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?!dense_35/MLCMatMul/ReadVariableOp?
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_51/Conv2D/ReadVariableOp?
conv2d_51/Conv2D	MLCConv2Dinputs'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
num_args *
paddingVALID*
strides
2
conv2d_51/Conv2D?
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp?
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_51/Relu?
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_52/Conv2D/ReadVariableOp?
conv2d_52/Conv2D	MLCConv2Dconv2d_51/Relu:activations:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv2d_52/Conv2D?
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp?
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_52/BiasAdd~
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_52/Relu?
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_53/Conv2D/ReadVariableOp?
conv2d_53/Conv2D	MLCConv2Dconv2d_52/Relu:activations:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
num_args *
paddingVALID*
strides
2
conv2d_53/Conv2D?
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp?
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_53/BiasAdd
conv2d_53/ReluReluconv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_53/Relu?
conv1d_17/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_17/conv1d/ExpandDims/dim?
conv1d_17/conv1d/ExpandDims
ExpandDimsconv2d_53/Relu:activations:0(conv1d_17/conv1d/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????2
conv1d_17/conv1d/ExpandDims?
,conv1d_17/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_17_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype02.
,conv1d_17/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_17/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_17/conv1d/ExpandDims_1/dim?
conv1d_17/conv1d/ExpandDims_1
ExpandDims4conv1d_17/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_17/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?2
conv1d_17/conv1d/ExpandDims_1?
conv1d_17/conv1d/ShapeShape$conv1d_17/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d_17/conv1d/Shape?
$conv1d_17/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_17/conv1d/strided_slice/stack?
&conv1d_17/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&conv1d_17/conv1d/strided_slice/stack_1?
&conv1d_17/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_17/conv1d/strided_slice/stack_2?
conv1d_17/conv1d/strided_sliceStridedSliceconv1d_17/conv1d/Shape:output:0-conv1d_17/conv1d/strided_slice/stack:output:0/conv1d_17/conv1d/strided_slice/stack_1:output:0/conv1d_17/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_17/conv1d/strided_slice?
conv1d_17/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      ?   2 
conv1d_17/conv1d/Reshape/shape?
conv1d_17/conv1d/ReshapeReshape$conv1d_17/conv1d/ExpandDims:output:0'conv1d_17/conv1d/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
conv1d_17/conv1d/Reshape?
conv1d_17/conv1d/Conv2DConv2D!conv1d_17/conv1d/Reshape:output:0&conv1d_17/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d_17/conv1d/Conv2D?
 conv1d_17/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2"
 conv1d_17/conv1d/concat/values_1?
conv1d_17/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d_17/conv1d/concat/axis?
conv1d_17/conv1d/concatConcatV2'conv1d_17/conv1d/strided_slice:output:0)conv1d_17/conv1d/concat/values_1:output:0%conv1d_17/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_17/conv1d/concat?
conv1d_17/conv1d/Reshape_1Reshape conv1d_17/conv1d/Conv2D:output:0 conv1d_17/conv1d/concat:output:0*
T0*3
_output_shapes!
:?????????2
conv1d_17/conv1d/Reshape_1?
conv1d_17/conv1d/SqueezeSqueeze#conv1d_17/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_17/conv1d/Squeeze?
"conv1d_17/squeeze_batch_dims/ShapeShape!conv1d_17/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2$
"conv1d_17/squeeze_batch_dims/Shape?
0conv1d_17/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv1d_17/squeeze_batch_dims/strided_slice/stack?
2conv1d_17/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????24
2conv1d_17/squeeze_batch_dims/strided_slice/stack_1?
2conv1d_17/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv1d_17/squeeze_batch_dims/strided_slice/stack_2?
*conv1d_17/squeeze_batch_dims/strided_sliceStridedSlice+conv1d_17/squeeze_batch_dims/Shape:output:09conv1d_17/squeeze_batch_dims/strided_slice/stack:output:0;conv1d_17/squeeze_batch_dims/strided_slice/stack_1:output:0;conv1d_17/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2,
*conv1d_17/squeeze_batch_dims/strided_slice?
*conv1d_17/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2,
*conv1d_17/squeeze_batch_dims/Reshape/shape?
$conv1d_17/squeeze_batch_dims/ReshapeReshape!conv1d_17/conv1d/Squeeze:output:03conv1d_17/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2&
$conv1d_17/squeeze_batch_dims/Reshape?
3conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp<conv1d_17_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOp?
$conv1d_17/squeeze_batch_dims/BiasAddBiasAdd-conv1d_17/squeeze_batch_dims/Reshape:output:0;conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2&
$conv1d_17/squeeze_batch_dims/BiasAdd?
,conv1d_17/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2.
,conv1d_17/squeeze_batch_dims/concat/values_1?
(conv1d_17/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(conv1d_17/squeeze_batch_dims/concat/axis?
#conv1d_17/squeeze_batch_dims/concatConcatV23conv1d_17/squeeze_batch_dims/strided_slice:output:05conv1d_17/squeeze_batch_dims/concat/values_1:output:01conv1d_17/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#conv1d_17/squeeze_batch_dims/concat?
&conv1d_17/squeeze_batch_dims/Reshape_1Reshape-conv1d_17/squeeze_batch_dims/BiasAdd:output:0,conv1d_17/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????2(
&conv1d_17/squeeze_batch_dims/Reshape_1?
conv1d_17/ReluRelu/conv1d_17/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????2
conv1d_17/Reluu
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_17/Const?
flatten_17/ReshapeReshapeconv1d_17/Relu:activations:0flatten_17/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_17/Reshape?
!dense_34/MLCMatMul/ReadVariableOpReadVariableOp*dense_34_mlcmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02#
!dense_34/MLCMatMul/ReadVariableOp?
dense_34/MLCMatMul	MLCMatMulflatten_17/Reshape:output:0)dense_34/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_34/MLCMatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MLCMatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_34/BiasAdds
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_34/Relu?
!dense_35/MLCMatMul/ReadVariableOpReadVariableOp*dense_35_mlcmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02#
!dense_35/MLCMatMul/ReadVariableOp?
dense_35/MLCMatMul	MLCMatMuldense_34/Relu:activations:0)dense_35/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_35/MLCMatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MLCMatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_35/BiasAdd|
dense_35/SigmoidSigmoiddense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_35/Sigmoid?
IdentityIdentitydense_35/Sigmoid:y:0-^conv1d_17/conv1d/ExpandDims_1/ReadVariableOp4^conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp"^dense_34/MLCMatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp"^dense_35/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::2\
,conv1d_17/conv1d/ExpandDims_1/ReadVariableOp,conv1d_17/conv1d/ExpandDims_1/ReadVariableOp2j
3conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOp3conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOp2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2F
!dense_34/MLCMatMul/ReadVariableOp!dense_34/MLCMatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2F
!dense_35/MLCMatMul/ReadVariableOp!dense_35/MLCMatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
E__inference_conv1d_17_layer_call_and_return_conditional_losses_425590

inputs/
+conv1d_expanddims_1_readvariableop_resource6
2squeeze_batch_dims_biasadd_readvariableop_resource
identity??"conv1d/ExpandDims_1/ReadVariableOp?)squeeze_batch_dims/BiasAdd/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?2
conv1d/ExpandDims_1f
conv1d/ShapeShapeconv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d/Shape?
conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
conv1d/strided_slice/stack?
conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2
conv1d/strided_slice/stack_1?
conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
conv1d/strided_slice/stack_2?
conv1d/strided_sliceStridedSliceconv1d/Shape:output:0#conv1d/strided_slice/stack:output:0%conv1d/strided_slice/stack_1:output:0%conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/strided_slice?
conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      ?   2
conv1d/Reshape/shape?
conv1d/ReshapeReshapeconv1d/ExpandDims:output:0conv1d/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
conv1d/Reshape?
conv1d/Conv2DConv2Dconv1d/Reshape:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d/Conv2D?
conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/concat/values_1s
conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/concat/axis?
conv1d/concatConcatV2conv1d/strided_slice:output:0conv1d/concat/values_1:output:0conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/concat?
conv1d/Reshape_1Reshapeconv1d/Conv2D:output:0conv1d/concat:output:0*
T0*3
_output_shapes!
:?????????2
conv1d/Reshape_1?
conv1d/SqueezeSqueezeconv1d/Reshape_1:output:0*
T0*/
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze{
squeeze_batch_dims/ShapeShapeconv1d/Squeeze:output:0*
T0*
_output_shapes
:2
squeeze_batch_dims/Shape?
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&squeeze_batch_dims/strided_slice/stack?
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2*
(squeeze_batch_dims/strided_slice/stack_1?
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(squeeze_batch_dims/strided_slice/stack_2?
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2"
 squeeze_batch_dims/strided_slice?
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2"
 squeeze_batch_dims/Reshape/shape?
squeeze_batch_dims/ReshapeReshapeconv1d/Squeeze:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
squeeze_batch_dims/Reshape?
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)squeeze_batch_dims/BiasAdd/ReadVariableOp?
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
squeeze_batch_dims/BiasAdd?
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2$
"squeeze_batch_dims/concat/values_1?
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
squeeze_batch_dims/concat/axis?
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2
squeeze_batch_dims/concat?
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????2
squeeze_batch_dims/Reshape_1u
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0#^conv1d/ExpandDims_1/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_425154

inputs
conv2d_51_425122
conv2d_51_425124
conv2d_52_425127
conv2d_52_425129
conv2d_53_425132
conv2d_53_425134
conv1d_17_425137
conv1d_17_425139
dense_34_425143
dense_34_425145
dense_35_425148
dense_35_425150
identity??!conv1d_17/StatefulPartitionedCall?!conv2d_51/StatefulPartitionedCall?!conv2d_52/StatefulPartitionedCall?!conv2d_53/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_51_425122conv2d_51_425124*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_4248882#
!conv2d_51/StatefulPartitionedCall?
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0conv2d_52_425127conv2d_52_425129*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_4249152#
!conv2d_52/StatefulPartitionedCall?
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0conv2d_53_425132conv2d_53_425134*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_53_layer_call_and_return_conditional_losses_4249422#
!conv2d_53/StatefulPartitionedCall?
!conv1d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0conv1d_17_425137conv1d_17_425139*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_17_layer_call_and_return_conditional_losses_4249962#
!conv1d_17/StatefulPartitionedCall?
flatten_17/PartitionedCallPartitionedCall*conv1d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_4250182
flatten_17/PartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_34_425143dense_34_425145*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_4250372"
 dense_34/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_425148dense_35_425150*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_4250642"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0"^conv1d_17/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::2F
!conv1d_17/StatefulPartitionedCall!conv1d_17/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_425081
input_18
conv2d_51_424899
conv2d_51_424901
conv2d_52_424926
conv2d_52_424928
conv2d_53_424953
conv2d_53_424955
conv1d_17_425007
conv1d_17_425009
dense_34_425048
dense_34_425050
dense_35_425075
dense_35_425077
identity??!conv1d_17/StatefulPartitionedCall?!conv2d_51/StatefulPartitionedCall?!conv2d_52/StatefulPartitionedCall?!conv2d_53/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCallinput_18conv2d_51_424899conv2d_51_424901*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_4248882#
!conv2d_51/StatefulPartitionedCall?
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0conv2d_52_424926conv2d_52_424928*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_4249152#
!conv2d_52/StatefulPartitionedCall?
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0conv2d_53_424953conv2d_53_424955*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_53_layer_call_and_return_conditional_losses_4249422#
!conv2d_53/StatefulPartitionedCall?
!conv1d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0conv1d_17_425007conv1d_17_425009*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_17_layer_call_and_return_conditional_losses_4249962#
!conv1d_17/StatefulPartitionedCall?
flatten_17/PartitionedCallPartitionedCall*conv1d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_4250182
flatten_17/PartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_34_425048dense_34_425050*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_4250372"
 dense_34/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_425075dense_35_425077*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_4250642"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0"^conv1d_17/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::2F
!conv1d_17/StatefulPartitionedCall!conv1d_17/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_18
?	
?
.__inference_sequential_17_layer_call_fn_425463

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_4251542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_flatten_17_layer_call_and_return_conditional_losses_425018

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_52_layer_call_and_return_conditional_losses_425523

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?l
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_425359

inputs,
(conv2d_51_conv2d_readvariableop_resource-
)conv2d_51_biasadd_readvariableop_resource,
(conv2d_52_conv2d_readvariableop_resource-
)conv2d_52_biasadd_readvariableop_resource,
(conv2d_53_conv2d_readvariableop_resource-
)conv2d_53_biasadd_readvariableop_resource9
5conv1d_17_conv1d_expanddims_1_readvariableop_resource@
<conv1d_17_squeeze_batch_dims_biasadd_readvariableop_resource.
*dense_34_mlcmatmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource.
*dense_35_mlcmatmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity??,conv1d_17/conv1d/ExpandDims_1/ReadVariableOp?3conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOp? conv2d_51/BiasAdd/ReadVariableOp?conv2d_51/Conv2D/ReadVariableOp? conv2d_52/BiasAdd/ReadVariableOp?conv2d_52/Conv2D/ReadVariableOp? conv2d_53/BiasAdd/ReadVariableOp?conv2d_53/Conv2D/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?!dense_34/MLCMatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?!dense_35/MLCMatMul/ReadVariableOp?
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_51/Conv2D/ReadVariableOp?
conv2d_51/Conv2D	MLCConv2Dinputs'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
num_args *
paddingVALID*
strides
2
conv2d_51/Conv2D?
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp?
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_51/Relu?
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_52/Conv2D/ReadVariableOp?
conv2d_52/Conv2D	MLCConv2Dconv2d_51/Relu:activations:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2
conv2d_52/Conv2D?
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp?
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_52/BiasAdd~
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_52/Relu?
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_53/Conv2D/ReadVariableOp?
conv2d_53/Conv2D	MLCConv2Dconv2d_52/Relu:activations:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
num_args *
paddingVALID*
strides
2
conv2d_53/Conv2D?
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp?
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_53/BiasAdd
conv2d_53/ReluReluconv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_53/Relu?
conv1d_17/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_17/conv1d/ExpandDims/dim?
conv1d_17/conv1d/ExpandDims
ExpandDimsconv2d_53/Relu:activations:0(conv1d_17/conv1d/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????2
conv1d_17/conv1d/ExpandDims?
,conv1d_17/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_17_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype02.
,conv1d_17/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_17/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_17/conv1d/ExpandDims_1/dim?
conv1d_17/conv1d/ExpandDims_1
ExpandDims4conv1d_17/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_17/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?2
conv1d_17/conv1d/ExpandDims_1?
conv1d_17/conv1d/ShapeShape$conv1d_17/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d_17/conv1d/Shape?
$conv1d_17/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_17/conv1d/strided_slice/stack?
&conv1d_17/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&conv1d_17/conv1d/strided_slice/stack_1?
&conv1d_17/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_17/conv1d/strided_slice/stack_2?
conv1d_17/conv1d/strided_sliceStridedSliceconv1d_17/conv1d/Shape:output:0-conv1d_17/conv1d/strided_slice/stack:output:0/conv1d_17/conv1d/strided_slice/stack_1:output:0/conv1d_17/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_17/conv1d/strided_slice?
conv1d_17/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      ?   2 
conv1d_17/conv1d/Reshape/shape?
conv1d_17/conv1d/ReshapeReshape$conv1d_17/conv1d/ExpandDims:output:0'conv1d_17/conv1d/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
conv1d_17/conv1d/Reshape?
conv1d_17/conv1d/Conv2DConv2D!conv1d_17/conv1d/Reshape:output:0&conv1d_17/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d_17/conv1d/Conv2D?
 conv1d_17/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2"
 conv1d_17/conv1d/concat/values_1?
conv1d_17/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d_17/conv1d/concat/axis?
conv1d_17/conv1d/concatConcatV2'conv1d_17/conv1d/strided_slice:output:0)conv1d_17/conv1d/concat/values_1:output:0%conv1d_17/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_17/conv1d/concat?
conv1d_17/conv1d/Reshape_1Reshape conv1d_17/conv1d/Conv2D:output:0 conv1d_17/conv1d/concat:output:0*
T0*3
_output_shapes!
:?????????2
conv1d_17/conv1d/Reshape_1?
conv1d_17/conv1d/SqueezeSqueeze#conv1d_17/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d_17/conv1d/Squeeze?
"conv1d_17/squeeze_batch_dims/ShapeShape!conv1d_17/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2$
"conv1d_17/squeeze_batch_dims/Shape?
0conv1d_17/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0conv1d_17/squeeze_batch_dims/strided_slice/stack?
2conv1d_17/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????24
2conv1d_17/squeeze_batch_dims/strided_slice/stack_1?
2conv1d_17/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2conv1d_17/squeeze_batch_dims/strided_slice/stack_2?
*conv1d_17/squeeze_batch_dims/strided_sliceStridedSlice+conv1d_17/squeeze_batch_dims/Shape:output:09conv1d_17/squeeze_batch_dims/strided_slice/stack:output:0;conv1d_17/squeeze_batch_dims/strided_slice/stack_1:output:0;conv1d_17/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2,
*conv1d_17/squeeze_batch_dims/strided_slice?
*conv1d_17/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2,
*conv1d_17/squeeze_batch_dims/Reshape/shape?
$conv1d_17/squeeze_batch_dims/ReshapeReshape!conv1d_17/conv1d/Squeeze:output:03conv1d_17/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2&
$conv1d_17/squeeze_batch_dims/Reshape?
3conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp<conv1d_17_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOp?
$conv1d_17/squeeze_batch_dims/BiasAddBiasAdd-conv1d_17/squeeze_batch_dims/Reshape:output:0;conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2&
$conv1d_17/squeeze_batch_dims/BiasAdd?
,conv1d_17/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2.
,conv1d_17/squeeze_batch_dims/concat/values_1?
(conv1d_17/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(conv1d_17/squeeze_batch_dims/concat/axis?
#conv1d_17/squeeze_batch_dims/concatConcatV23conv1d_17/squeeze_batch_dims/strided_slice:output:05conv1d_17/squeeze_batch_dims/concat/values_1:output:01conv1d_17/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#conv1d_17/squeeze_batch_dims/concat?
&conv1d_17/squeeze_batch_dims/Reshape_1Reshape-conv1d_17/squeeze_batch_dims/BiasAdd:output:0,conv1d_17/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????2(
&conv1d_17/squeeze_batch_dims/Reshape_1?
conv1d_17/ReluRelu/conv1d_17/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????2
conv1d_17/Reluu
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_17/Const?
flatten_17/ReshapeReshapeconv1d_17/Relu:activations:0flatten_17/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_17/Reshape?
!dense_34/MLCMatMul/ReadVariableOpReadVariableOp*dense_34_mlcmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02#
!dense_34/MLCMatMul/ReadVariableOp?
dense_34/MLCMatMul	MLCMatMulflatten_17/Reshape:output:0)dense_34/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_34/MLCMatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MLCMatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_34/BiasAdds
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_34/Relu?
!dense_35/MLCMatMul/ReadVariableOpReadVariableOp*dense_35_mlcmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02#
!dense_35/MLCMatMul/ReadVariableOp?
dense_35/MLCMatMul	MLCMatMuldense_34/Relu:activations:0)dense_35/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_35/MLCMatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MLCMatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_35/BiasAdd|
dense_35/SigmoidSigmoiddense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_35/Sigmoid?
IdentityIdentitydense_35/Sigmoid:y:0-^conv1d_17/conv1d/ExpandDims_1/ReadVariableOp4^conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp"^dense_34/MLCMatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp"^dense_35/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::2\
,conv1d_17/conv1d/ExpandDims_1/ReadVariableOp,conv1d_17/conv1d/ExpandDims_1/ReadVariableOp2j
3conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOp3conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOp2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2F
!dense_34/MLCMatMul/ReadVariableOp!dense_34/MLCMatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2F
!dense_35/MLCMatMul/ReadVariableOp!dense_35/MLCMatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
D__inference_dense_35_layer_call_and_return_conditional_losses_425641

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
E__inference_conv2d_53_layer_call_and_return_conditional_losses_425543

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
num_args *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
E__inference_conv2d_51_layer_call_and_return_conditional_losses_424888

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
num_args *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_flatten_17_layer_call_fn_425610

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_4250182
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
~
)__inference_dense_35_layer_call_fn_425650

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_4250642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

*__inference_conv1d_17_layer_call_fn_425599

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1d_17_layer_call_and_return_conditional_losses_4249962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_51_layer_call_and_return_conditional_losses_425503

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
num_args *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
E__inference_conv1d_17_layer_call_and_return_conditional_losses_424996

inputs/
+conv1d_expanddims_1_readvariableop_resource6
2squeeze_batch_dims_biasadd_readvariableop_resource
identity??"conv1d/ExpandDims_1/ReadVariableOp?)squeeze_batch_dims/BiasAdd/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?2
conv1d/ExpandDims_1f
conv1d/ShapeShapeconv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d/Shape?
conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
conv1d/strided_slice/stack?
conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2
conv1d/strided_slice/stack_1?
conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
conv1d/strided_slice/stack_2?
conv1d/strided_sliceStridedSliceconv1d/Shape:output:0#conv1d/strided_slice/stack:output:0%conv1d/strided_slice/stack_1:output:0%conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/strided_slice?
conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      ?   2
conv1d/Reshape/shape?
conv1d/ReshapeReshapeconv1d/ExpandDims:output:0conv1d/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
conv1d/Reshape?
conv1d/Conv2DConv2Dconv1d/Reshape:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d/Conv2D?
conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/concat/values_1s
conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/concat/axis?
conv1d/concatConcatV2conv1d/strided_slice:output:0conv1d/concat/values_1:output:0conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/concat?
conv1d/Reshape_1Reshapeconv1d/Conv2D:output:0conv1d/concat:output:0*
T0*3
_output_shapes!
:?????????2
conv1d/Reshape_1?
conv1d/SqueezeSqueezeconv1d/Reshape_1:output:0*
T0*/
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze{
squeeze_batch_dims/ShapeShapeconv1d/Squeeze:output:0*
T0*
_output_shapes
:2
squeeze_batch_dims/Shape?
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&squeeze_batch_dims/strided_slice/stack?
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2*
(squeeze_batch_dims/strided_slice/stack_1?
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(squeeze_batch_dims/strided_slice/stack_2?
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2"
 squeeze_batch_dims/strided_slice?
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2"
 squeeze_batch_dims/Reshape/shape?
squeeze_batch_dims/ReshapeReshapeconv1d/Squeeze:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
squeeze_batch_dims/Reshape?
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)squeeze_batch_dims/BiasAdd/ReadVariableOp?
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
squeeze_batch_dims/BiasAdd?
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2$
"squeeze_batch_dims/concat/values_1?
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
squeeze_batch_dims/concat/axis?
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2
squeeze_batch_dims/concat?
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????2
squeeze_batch_dims/Reshape_1u
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0#^conv1d/ExpandDims_1/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_conv2d_52_layer_call_fn_425532

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_4249152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
D__inference_dense_34_layer_call_and_return_conditional_losses_425037

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense_35_layer_call_and_return_conditional_losses_425064

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_424873
input_18:
6sequential_17_conv2d_51_conv2d_readvariableop_resource;
7sequential_17_conv2d_51_biasadd_readvariableop_resource:
6sequential_17_conv2d_52_conv2d_readvariableop_resource;
7sequential_17_conv2d_52_biasadd_readvariableop_resource:
6sequential_17_conv2d_53_conv2d_readvariableop_resource;
7sequential_17_conv2d_53_biasadd_readvariableop_resourceG
Csequential_17_conv1d_17_conv1d_expanddims_1_readvariableop_resourceN
Jsequential_17_conv1d_17_squeeze_batch_dims_biasadd_readvariableop_resource<
8sequential_17_dense_34_mlcmatmul_readvariableop_resource:
6sequential_17_dense_34_biasadd_readvariableop_resource<
8sequential_17_dense_35_mlcmatmul_readvariableop_resource:
6sequential_17_dense_35_biasadd_readvariableop_resource
identity??:sequential_17/conv1d_17/conv1d/ExpandDims_1/ReadVariableOp?Asequential_17/conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOp?.sequential_17/conv2d_51/BiasAdd/ReadVariableOp?-sequential_17/conv2d_51/Conv2D/ReadVariableOp?.sequential_17/conv2d_52/BiasAdd/ReadVariableOp?-sequential_17/conv2d_52/Conv2D/ReadVariableOp?.sequential_17/conv2d_53/BiasAdd/ReadVariableOp?-sequential_17/conv2d_53/Conv2D/ReadVariableOp?-sequential_17/dense_34/BiasAdd/ReadVariableOp?/sequential_17/dense_34/MLCMatMul/ReadVariableOp?-sequential_17/dense_35/BiasAdd/ReadVariableOp?/sequential_17/dense_35/MLCMatMul/ReadVariableOp?
-sequential_17/conv2d_51/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_17/conv2d_51/Conv2D/ReadVariableOp?
sequential_17/conv2d_51/Conv2D	MLCConv2Dinput_185sequential_17/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
num_args *
paddingVALID*
strides
2 
sequential_17/conv2d_51/Conv2D?
.sequential_17/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_17/conv2d_51/BiasAdd/ReadVariableOp?
sequential_17/conv2d_51/BiasAddBiasAdd'sequential_17/conv2d_51/Conv2D:output:06sequential_17/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2!
sequential_17/conv2d_51/BiasAdd?
sequential_17/conv2d_51/ReluRelu(sequential_17/conv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_17/conv2d_51/Relu?
-sequential_17/conv2d_52/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02/
-sequential_17/conv2d_52/Conv2D/ReadVariableOp?
sequential_17/conv2d_52/Conv2D	MLCConv2D*sequential_17/conv2d_51/Relu:activations:05sequential_17/conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
num_args *
paddingVALID*
strides
2 
sequential_17/conv2d_52/Conv2D?
.sequential_17/conv2d_52/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_52_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_17/conv2d_52/BiasAdd/ReadVariableOp?
sequential_17/conv2d_52/BiasAddBiasAdd'sequential_17/conv2d_52/Conv2D:output:06sequential_17/conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2!
sequential_17/conv2d_52/BiasAdd?
sequential_17/conv2d_52/ReluRelu(sequential_17/conv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_17/conv2d_52/Relu?
-sequential_17/conv2d_53/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_53_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02/
-sequential_17/conv2d_53/Conv2D/ReadVariableOp?
sequential_17/conv2d_53/Conv2D	MLCConv2D*sequential_17/conv2d_52/Relu:activations:05sequential_17/conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
num_args *
paddingVALID*
strides
2 
sequential_17/conv2d_53/Conv2D?
.sequential_17/conv2d_53/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential_17/conv2d_53/BiasAdd/ReadVariableOp?
sequential_17/conv2d_53/BiasAddBiasAdd'sequential_17/conv2d_53/Conv2D:output:06sequential_17/conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2!
sequential_17/conv2d_53/BiasAdd?
sequential_17/conv2d_53/ReluRelu(sequential_17/conv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential_17/conv2d_53/Relu?
-sequential_17/conv1d_17/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_17/conv1d_17/conv1d/ExpandDims/dim?
)sequential_17/conv1d_17/conv1d/ExpandDims
ExpandDims*sequential_17/conv2d_53/Relu:activations:06sequential_17/conv1d_17/conv1d/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :??????????2+
)sequential_17/conv1d_17/conv1d/ExpandDims?
:sequential_17/conv1d_17/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCsequential_17_conv1d_17_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype02<
:sequential_17/conv1d_17/conv1d/ExpandDims_1/ReadVariableOp?
/sequential_17/conv1d_17/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_17/conv1d_17/conv1d/ExpandDims_1/dim?
+sequential_17/conv1d_17/conv1d/ExpandDims_1
ExpandDimsBsequential_17/conv1d_17/conv1d/ExpandDims_1/ReadVariableOp:value:08sequential_17/conv1d_17/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?2-
+sequential_17/conv1d_17/conv1d/ExpandDims_1?
$sequential_17/conv1d_17/conv1d/ShapeShape2sequential_17/conv1d_17/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2&
$sequential_17/conv1d_17/conv1d/Shape?
2sequential_17/conv1d_17/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_17/conv1d_17/conv1d/strided_slice/stack?
4sequential_17/conv1d_17/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????26
4sequential_17/conv1d_17/conv1d/strided_slice/stack_1?
4sequential_17/conv1d_17/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_17/conv1d_17/conv1d/strided_slice/stack_2?
,sequential_17/conv1d_17/conv1d/strided_sliceStridedSlice-sequential_17/conv1d_17/conv1d/Shape:output:0;sequential_17/conv1d_17/conv1d/strided_slice/stack:output:0=sequential_17/conv1d_17/conv1d/strided_slice/stack_1:output:0=sequential_17/conv1d_17/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2.
,sequential_17/conv1d_17/conv1d/strided_slice?
,sequential_17/conv1d_17/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      ?   2.
,sequential_17/conv1d_17/conv1d/Reshape/shape?
&sequential_17/conv1d_17/conv1d/ReshapeReshape2sequential_17/conv1d_17/conv1d/ExpandDims:output:05sequential_17/conv1d_17/conv1d/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2(
&sequential_17/conv1d_17/conv1d/Reshape?
%sequential_17/conv1d_17/conv1d/Conv2DConv2D/sequential_17/conv1d_17/conv1d/Reshape:output:04sequential_17/conv1d_17/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2'
%sequential_17/conv1d_17/conv1d/Conv2D?
.sequential_17/conv1d_17/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         20
.sequential_17/conv1d_17/conv1d/concat/values_1?
*sequential_17/conv1d_17/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*sequential_17/conv1d_17/conv1d/concat/axis?
%sequential_17/conv1d_17/conv1d/concatConcatV25sequential_17/conv1d_17/conv1d/strided_slice:output:07sequential_17/conv1d_17/conv1d/concat/values_1:output:03sequential_17/conv1d_17/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_17/conv1d_17/conv1d/concat?
(sequential_17/conv1d_17/conv1d/Reshape_1Reshape.sequential_17/conv1d_17/conv1d/Conv2D:output:0.sequential_17/conv1d_17/conv1d/concat:output:0*
T0*3
_output_shapes!
:?????????2*
(sequential_17/conv1d_17/conv1d/Reshape_1?
&sequential_17/conv1d_17/conv1d/SqueezeSqueeze1sequential_17/conv1d_17/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:?????????*
squeeze_dims

?????????2(
&sequential_17/conv1d_17/conv1d/Squeeze?
0sequential_17/conv1d_17/squeeze_batch_dims/ShapeShape/sequential_17/conv1d_17/conv1d/Squeeze:output:0*
T0*
_output_shapes
:22
0sequential_17/conv1d_17/squeeze_batch_dims/Shape?
>sequential_17/conv1d_17/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_17/conv1d_17/squeeze_batch_dims/strided_slice/stack?
@sequential_17/conv1d_17/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2B
@sequential_17/conv1d_17/squeeze_batch_dims/strided_slice/stack_1?
@sequential_17/conv1d_17/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_17/conv1d_17/squeeze_batch_dims/strided_slice/stack_2?
8sequential_17/conv1d_17/squeeze_batch_dims/strided_sliceStridedSlice9sequential_17/conv1d_17/squeeze_batch_dims/Shape:output:0Gsequential_17/conv1d_17/squeeze_batch_dims/strided_slice/stack:output:0Isequential_17/conv1d_17/squeeze_batch_dims/strided_slice/stack_1:output:0Isequential_17/conv1d_17/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2:
8sequential_17/conv1d_17/squeeze_batch_dims/strided_slice?
8sequential_17/conv1d_17/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2:
8sequential_17/conv1d_17/squeeze_batch_dims/Reshape/shape?
2sequential_17/conv1d_17/squeeze_batch_dims/ReshapeReshape/sequential_17/conv1d_17/conv1d/Squeeze:output:0Asequential_17/conv1d_17/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????24
2sequential_17/conv1d_17/squeeze_batch_dims/Reshape?
Asequential_17/conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpJsequential_17_conv1d_17_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02C
Asequential_17/conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOp?
2sequential_17/conv1d_17/squeeze_batch_dims/BiasAddBiasAdd;sequential_17/conv1d_17/squeeze_batch_dims/Reshape:output:0Isequential_17/conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????24
2sequential_17/conv1d_17/squeeze_batch_dims/BiasAdd?
:sequential_17/conv1d_17/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2<
:sequential_17/conv1d_17/squeeze_batch_dims/concat/values_1?
6sequential_17/conv1d_17/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6sequential_17/conv1d_17/squeeze_batch_dims/concat/axis?
1sequential_17/conv1d_17/squeeze_batch_dims/concatConcatV2Asequential_17/conv1d_17/squeeze_batch_dims/strided_slice:output:0Csequential_17/conv1d_17/squeeze_batch_dims/concat/values_1:output:0?sequential_17/conv1d_17/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:23
1sequential_17/conv1d_17/squeeze_batch_dims/concat?
4sequential_17/conv1d_17/squeeze_batch_dims/Reshape_1Reshape;sequential_17/conv1d_17/squeeze_batch_dims/BiasAdd:output:0:sequential_17/conv1d_17/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:?????????26
4sequential_17/conv1d_17/squeeze_batch_dims/Reshape_1?
sequential_17/conv1d_17/ReluRelu=sequential_17/conv1d_17/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:?????????2
sequential_17/conv1d_17/Relu?
sequential_17/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2 
sequential_17/flatten_17/Const?
 sequential_17/flatten_17/ReshapeReshape*sequential_17/conv1d_17/Relu:activations:0'sequential_17/flatten_17/Const:output:0*
T0*(
_output_shapes
:??????????2"
 sequential_17/flatten_17/Reshape?
/sequential_17/dense_34/MLCMatMul/ReadVariableOpReadVariableOp8sequential_17_dense_34_mlcmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype021
/sequential_17/dense_34/MLCMatMul/ReadVariableOp?
 sequential_17/dense_34/MLCMatMul	MLCMatMul)sequential_17/flatten_17/Reshape:output:07sequential_17/dense_34/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2"
 sequential_17/dense_34/MLCMatMul?
-sequential_17/dense_34/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_17/dense_34/BiasAdd/ReadVariableOp?
sequential_17/dense_34/BiasAddBiasAdd*sequential_17/dense_34/MLCMatMul:product:05sequential_17/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2 
sequential_17/dense_34/BiasAdd?
sequential_17/dense_34/ReluRelu'sequential_17/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
sequential_17/dense_34/Relu?
/sequential_17/dense_35/MLCMatMul/ReadVariableOpReadVariableOp8sequential_17_dense_35_mlcmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype021
/sequential_17/dense_35/MLCMatMul/ReadVariableOp?
 sequential_17/dense_35/MLCMatMul	MLCMatMul)sequential_17/dense_34/Relu:activations:07sequential_17/dense_35/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2"
 sequential_17/dense_35/MLCMatMul?
-sequential_17/dense_35/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential_17/dense_35/BiasAdd/ReadVariableOp?
sequential_17/dense_35/BiasAddBiasAdd*sequential_17/dense_35/MLCMatMul:product:05sequential_17/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
sequential_17/dense_35/BiasAdd?
sequential_17/dense_35/SigmoidSigmoid'sequential_17/dense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2 
sequential_17/dense_35/Sigmoid?
IdentityIdentity"sequential_17/dense_35/Sigmoid:y:0;^sequential_17/conv1d_17/conv1d/ExpandDims_1/ReadVariableOpB^sequential_17/conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOp/^sequential_17/conv2d_51/BiasAdd/ReadVariableOp.^sequential_17/conv2d_51/Conv2D/ReadVariableOp/^sequential_17/conv2d_52/BiasAdd/ReadVariableOp.^sequential_17/conv2d_52/Conv2D/ReadVariableOp/^sequential_17/conv2d_53/BiasAdd/ReadVariableOp.^sequential_17/conv2d_53/Conv2D/ReadVariableOp.^sequential_17/dense_34/BiasAdd/ReadVariableOp0^sequential_17/dense_34/MLCMatMul/ReadVariableOp.^sequential_17/dense_35/BiasAdd/ReadVariableOp0^sequential_17/dense_35/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::2x
:sequential_17/conv1d_17/conv1d/ExpandDims_1/ReadVariableOp:sequential_17/conv1d_17/conv1d/ExpandDims_1/ReadVariableOp2?
Asequential_17/conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOpAsequential_17/conv1d_17/squeeze_batch_dims/BiasAdd/ReadVariableOp2`
.sequential_17/conv2d_51/BiasAdd/ReadVariableOp.sequential_17/conv2d_51/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_51/Conv2D/ReadVariableOp-sequential_17/conv2d_51/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_52/BiasAdd/ReadVariableOp.sequential_17/conv2d_52/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_52/Conv2D/ReadVariableOp-sequential_17/conv2d_52/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_53/BiasAdd/ReadVariableOp.sequential_17/conv2d_53/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_53/Conv2D/ReadVariableOp-sequential_17/conv2d_53/Conv2D/ReadVariableOp2^
-sequential_17/dense_34/BiasAdd/ReadVariableOp-sequential_17/dense_34/BiasAdd/ReadVariableOp2b
/sequential_17/dense_34/MLCMatMul/ReadVariableOp/sequential_17/dense_34/MLCMatMul/ReadVariableOp2^
-sequential_17/dense_35/BiasAdd/ReadVariableOp-sequential_17/dense_35/BiasAdd/ReadVariableOp2b
/sequential_17/dense_35/MLCMatMul/ReadVariableOp/sequential_17/dense_35/MLCMatMul/ReadVariableOp:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_18
ڽ
?
"__inference__traced_restore_425953
file_prefix%
!assignvariableop_conv2d_51_kernel%
!assignvariableop_1_conv2d_51_bias'
#assignvariableop_2_conv2d_52_kernel%
!assignvariableop_3_conv2d_52_bias'
#assignvariableop_4_conv2d_53_kernel%
!assignvariableop_5_conv2d_53_bias'
#assignvariableop_6_conv1d_17_kernel%
!assignvariableop_7_conv1d_17_bias&
"assignvariableop_8_dense_34_kernel$
 assignvariableop_9_dense_34_bias'
#assignvariableop_10_dense_35_kernel%
!assignvariableop_11_dense_35_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_1/
+assignvariableop_21_adam_conv2d_51_kernel_m-
)assignvariableop_22_adam_conv2d_51_bias_m/
+assignvariableop_23_adam_conv2d_52_kernel_m-
)assignvariableop_24_adam_conv2d_52_bias_m/
+assignvariableop_25_adam_conv2d_53_kernel_m-
)assignvariableop_26_adam_conv2d_53_bias_m/
+assignvariableop_27_adam_conv1d_17_kernel_m-
)assignvariableop_28_adam_conv1d_17_bias_m.
*assignvariableop_29_adam_dense_34_kernel_m,
(assignvariableop_30_adam_dense_34_bias_m.
*assignvariableop_31_adam_dense_35_kernel_m,
(assignvariableop_32_adam_dense_35_bias_m/
+assignvariableop_33_adam_conv2d_51_kernel_v-
)assignvariableop_34_adam_conv2d_51_bias_v/
+assignvariableop_35_adam_conv2d_52_kernel_v-
)assignvariableop_36_adam_conv2d_52_bias_v/
+assignvariableop_37_adam_conv2d_53_kernel_v-
)assignvariableop_38_adam_conv2d_53_bias_v/
+assignvariableop_39_adam_conv1d_17_kernel_v-
)assignvariableop_40_adam_conv1d_17_bias_v.
*assignvariableop_41_adam_dense_34_kernel_v,
(assignvariableop_42_adam_dense_34_bias_v.
*assignvariableop_43_adam_dense_35_kernel_v,
(assignvariableop_44_adam_dense_35_bias_v
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_51_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_51_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_52_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_52_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_53_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_53_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv1d_17_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv1d_17_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_34_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_34_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_35_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_35_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_51_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_51_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_52_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_52_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_53_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_53_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv1d_17_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv1d_17_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_34_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_34_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_35_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_35_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_51_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_51_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_52_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_52_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_53_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_53_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv1d_17_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv1d_17_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_34_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_34_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_35_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_35_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45?
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_189
serving_default_input_18:0?????????<
dense_350
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?I
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?F
_tf_keras_sequential?E{"class_name": "Sequential", "name": "sequential_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_18"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_51", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_52", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_53", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv1D", "config": {"name": "conv1d_17", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 10, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_18"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_51", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_52", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_53", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv1D", "config": {"name": "conv1d_17", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 10, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "loss", "from_logits": false}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_51", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
?	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_52", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 26, 32]}}
?	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_53", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 64]}}
?	

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_17", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 22, 128]}}
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 484}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 484]}}
?

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 10, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
6iter

7beta_1

8beta_2
	9decay
:learning_ratemnmompmqmrms mt!mu*mv+mw0mx1myvzv{v|v}v~v v?!v?*v?+v?0v?1v?"
	optimizer
v
0
1
2
3
4
5
 6
!7
*8
+9
010
111"
trackable_list_wrapper
v
0
1
2
3
4
5
 6
!7
*8
+9
010
111"
trackable_list_wrapper
 "
trackable_list_wrapper
?
;non_trainable_variables
<layer_metrics
=metrics

>layers
		variables
?layer_regularization_losses

trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:( 2conv2d_51/kernel
: 2conv2d_51/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
@non_trainable_variables
Alayer_metrics
Bmetrics

Clayers
	variables
Dlayer_regularization_losses
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_52/kernel
:@2conv2d_52/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Enon_trainable_variables
Flayer_metrics
Gmetrics

Hlayers
	variables
Ilayer_regularization_losses
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)@?2conv2d_53/kernel
:?2conv2d_53/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Jnon_trainable_variables
Klayer_metrics
Lmetrics

Mlayers
	variables
Nlayer_regularization_losses
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%?2conv1d_17/kernel
:2conv1d_17/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Onon_trainable_variables
Player_metrics
Qmetrics

Rlayers
"	variables
Slayer_regularization_losses
#trainable_variables
$regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tnon_trainable_variables
Ulayer_metrics
Vmetrics

Wlayers
&	variables
Xlayer_regularization_losses
'trainable_variables
(regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?@2dense_34/kernel
:@2dense_34/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ynon_trainable_variables
Zlayer_metrics
[metrics

\layers
,	variables
]layer_regularization_losses
-trainable_variables
.regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@
2dense_35/kernel
:
2dense_35/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
^non_trainable_variables
_layer_metrics
`metrics

alayers
2	variables
blayer_regularization_losses
3trainable_variables
4regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
c0
d1"
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
?
	etotal
	fcount
g	variables
h	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	itotal
	jcount
k
_fn_kwargs
l	variables
m	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
e0
f1"
trackable_list_wrapper
-
g	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
i0
j1"
trackable_list_wrapper
-
l	variables"
_generic_user_object
/:- 2Adam/conv2d_51/kernel/m
!: 2Adam/conv2d_51/bias/m
/:- @2Adam/conv2d_52/kernel/m
!:@2Adam/conv2d_52/bias/m
0:.@?2Adam/conv2d_53/kernel/m
": ?2Adam/conv2d_53/bias/m
,:*?2Adam/conv1d_17/kernel/m
!:2Adam/conv1d_17/bias/m
':%	?@2Adam/dense_34/kernel/m
 :@2Adam/dense_34/bias/m
&:$@
2Adam/dense_35/kernel/m
 :
2Adam/dense_35/bias/m
/:- 2Adam/conv2d_51/kernel/v
!: 2Adam/conv2d_51/bias/v
/:- @2Adam/conv2d_52/kernel/v
!:@2Adam/conv2d_52/bias/v
0:.@?2Adam/conv2d_53/kernel/v
": ?2Adam/conv2d_53/bias/v
,:*?2Adam/conv1d_17/kernel/v
!:2Adam/conv1d_17/bias/v
':%	?@2Adam/dense_34/kernel/v
 :@2Adam/dense_34/bias/v
&:$@
2Adam/dense_35/kernel/v
 :
2Adam/dense_35/bias/v
?2?
.__inference_sequential_17_layer_call_fn_425245
.__inference_sequential_17_layer_call_fn_425492
.__inference_sequential_17_layer_call_fn_425181
.__inference_sequential_17_layer_call_fn_425463?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_424873?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
input_18?????????
?2?
I__inference_sequential_17_layer_call_and_return_conditional_losses_425359
I__inference_sequential_17_layer_call_and_return_conditional_losses_425434
I__inference_sequential_17_layer_call_and_return_conditional_losses_425116
I__inference_sequential_17_layer_call_and_return_conditional_losses_425081?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv2d_51_layer_call_fn_425512?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_51_layer_call_and_return_conditional_losses_425503?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_52_layer_call_fn_425532?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_52_layer_call_and_return_conditional_losses_425523?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_53_layer_call_fn_425552?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_53_layer_call_and_return_conditional_losses_425543?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv1d_17_layer_call_fn_425599?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv1d_17_layer_call_and_return_conditional_losses_425590?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_flatten_17_layer_call_fn_425610?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_flatten_17_layer_call_and_return_conditional_losses_425605?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_34_layer_call_fn_425630?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_34_layer_call_and_return_conditional_losses_425621?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_35_layer_call_fn_425650?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_35_layer_call_and_return_conditional_losses_425641?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_425284input_18"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_424873~ !*+019?6
/?,
*?'
input_18?????????
? "3?0
.
dense_35"?
dense_35?????????
?
E__inference_conv1d_17_layer_call_and_return_conditional_losses_425590m !8?5
.?+
)?&
inputs??????????
? "-?*
#? 
0?????????
? ?
*__inference_conv1d_17_layer_call_fn_425599` !8?5
.?+
)?&
inputs??????????
? " ???????????
E__inference_conv2d_51_layer_call_and_return_conditional_losses_425503l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
*__inference_conv2d_51_layer_call_fn_425512_7?4
-?*
(?%
inputs?????????
? " ?????????? ?
E__inference_conv2d_52_layer_call_and_return_conditional_losses_425523l7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
*__inference_conv2d_52_layer_call_fn_425532_7?4
-?*
(?%
inputs????????? 
? " ??????????@?
E__inference_conv2d_53_layer_call_and_return_conditional_losses_425543m7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
*__inference_conv2d_53_layer_call_fn_425552`7?4
-?*
(?%
inputs?????????@
? "!????????????
D__inference_dense_34_layer_call_and_return_conditional_losses_425621]*+0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? }
)__inference_dense_34_layer_call_fn_425630P*+0?-
&?#
!?
inputs??????????
? "??????????@?
D__inference_dense_35_layer_call_and_return_conditional_losses_425641\01/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????

? |
)__inference_dense_35_layer_call_fn_425650O01/?,
%?"
 ?
inputs?????????@
? "??????????
?
F__inference_flatten_17_layer_call_and_return_conditional_losses_425605a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
+__inference_flatten_17_layer_call_fn_425610T7?4
-?*
(?%
inputs?????????
? "????????????
I__inference_sequential_17_layer_call_and_return_conditional_losses_425081x !*+01A?>
7?4
*?'
input_18?????????
p

 
? "%?"
?
0?????????

? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_425116x !*+01A?>
7?4
*?'
input_18?????????
p 

 
? "%?"
?
0?????????

? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_425359v !*+01??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????

? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_425434v !*+01??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
.__inference_sequential_17_layer_call_fn_425181k !*+01A?>
7?4
*?'
input_18?????????
p

 
? "??????????
?
.__inference_sequential_17_layer_call_fn_425245k !*+01A?>
7?4
*?'
input_18?????????
p 

 
? "??????????
?
.__inference_sequential_17_layer_call_fn_425463i !*+01??<
5?2
(?%
inputs?????????
p

 
? "??????????
?
.__inference_sequential_17_layer_call_fn_425492i !*+01??<
5?2
(?%
inputs?????????
p 

 
? "??????????
?
$__inference_signature_wrapper_425284? !*+01E?B
? 
;?8
6
input_18*?'
input_18?????????"3?0
.
dense_35"?
dense_35?????????
