use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::slice;

pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        // let get_tensor: impl Fn(&str) -> Tensor<f32> = |name: &str| {
        // ...    
        // };
        
        // LLamaParams {
        //     embedding_table: get_tensor(...),
        //     ...
        // }
        let layers = config.num_hidden_layers;

        // 打印所有张量的名称
        let get_tensor = |name: &str| {
            safetensor.tensor(name)
                .map(|tensor_view| {
                    let tensor_data = unsafe {
                        slice::from_raw_parts(
                            tensor_view.data().as_ptr() as *const f32,
                            tensor_view.shape().iter().product::<usize>(),
                        )
                    };
                    Tensor::new(tensor_data.to_vec(), &tensor_view.shape().to_vec())
                })
                .unwrap_or_else(|_| panic!("Failed to load tensor: {}", name))
        };

        let load_layer_tensors = |pattern: &str| {
            (0..layers)
            .map(|i| get_tensor(&format!("{}", pattern.replace("{}", &i.to_string())))) 
            .collect()
        };

        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: load_layer_tensors("model.layers.{}.input_layernorm.weight"),
            wq: load_layer_tensors("model.layers.{}.self_attn.q_proj.weight"),
            wk: load_layer_tensors("model.layers.{}.self_attn.k_proj.weight"),
            wv: load_layer_tensors("model.layers.{}.self_attn.v_proj.weight"),
            wo: load_layer_tensors("model.layers.{}.self_attn.o_proj.weight"),
            rms_ffn_w: load_layer_tensors("model.layers.{}.post_attention_layernorm.weight"),
            w_up: load_layer_tensors("model.layers.{}.mlp.up_proj.weight"),
            w_gate: load_layer_tensors("model.layers.{}.mlp.gate_proj.weight"),
            w_down: load_layer_tensors("model.layers.{}.mlp.down_proj.weight"),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
