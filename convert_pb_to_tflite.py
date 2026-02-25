import tensorflow as tf
import os
import argparse

def convert_pb_to_tflite(pb_path, tflite_path, input_nodes, output_nodes, input_shapes, optimize=True):
    """
    Frozen Graph (.pb) を TFLite に変換する。
    """
    print(f"Loading Frozen Graph: {pb_path}")
    
    if not os.path.exists(pb_path):
        print(f"Error: {pb_path} not found.")
        return

    # TFLiteConverter の初期化
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=pb_path,
        input_arrays=input_nodes,
        output_arrays=output_nodes,
        input_shapes=input_shapes
    )

    if optimize:
        print("Enabling float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    # 必要に応じて一部の演算で TensorFlow ops を許可
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.allow_custom_ops = True

    print("Converting... (This might take a while)")
    try:
        tflite_model = converter.convert()
        
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        
        print(f"Successfully converted to: {tflite_path}")
        print(f"File size: {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"Conversion failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TensorFlow Frozen Graph (.pb) to TFLite.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .pb file")
    parser.add_argument("--output", type=str, default="model.tflite", help="Path to the output .tflite file")
    
    # SSD MobileNet V2 等の一般的なデフォルト値
    parser.add_argument("--input_nodes", type=str, default="image_tensor", help="Input node names (comma separated)")
    parser.add_argument("--output_nodes", type=str, 
                        default="detection_boxes,detection_classes,detection_scores,num_detections", 
                        help="Output node names (comma separated)")
    parser.add_argument("--input_shape", type=str, default="1,300,300,3", help="Input shape (e.g. 1,300,300,3)")
    parser.add_argument("--no_opt", action="store_true", help="Disable float16 optimization")

    args = parser.parse_args()

    # 文字列引数をリストや辞書に変換
    input_nodes_list = [n.strip() for n in args.input_nodes.split(",")]
    output_nodes_list = [n.strip() for n in args.output_nodes.split(",")]
    
    shape_list = [int(s.strip()) for s in args.input_shape.split(",")]
    input_shapes = {input_nodes_list[0]: shape_list}

    convert_pb_to_tflite(
        args.input, 
        args.output, 
        input_nodes_list, 
        output_nodes_list, 
        input_shapes,
        optimize=not args.no_opt
    )
