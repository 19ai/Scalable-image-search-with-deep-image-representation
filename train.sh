source activate ./vir_env/
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH && export PATH=/usr/local/cuda-8.0/bin:$PATH && export LD_LIBRARY_PATH=cuda/lib64:$LD_LIBRARY_PATH && export PATH=cuda/bin:$PATH && export CUDNN_HOME=cuda && export C_INCLUDE_PATH=$CUDNN_HOME/include:$C_INCLUDE_PATH && export CPATH=$CUDNN_HOME/include:$CPATH && export LIBRARY_PATH=$CUDNN_HOME/lib64:$LIBRARY_PATH