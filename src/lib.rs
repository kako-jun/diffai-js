use diffai_core::{
    diff as core_diff, diff_paths as core_diff_paths, format_output as core_format_output,
    DiffOptions, DiffResult, OutputFormat, TensorStats,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use regex::Regex;

#[napi(object)]
pub struct JsDiffOptions {
    /// Numerical comparison tolerance
    pub epsilon: Option<f64>,

    /// Key to use for array element identification
    pub array_id_key: Option<String>,

    /// Regex pattern for keys to ignore
    pub ignore_keys_regex: Option<String>,

    /// Only show differences in paths containing this string
    pub path_filter: Option<String>,

    /// Output format
    pub output_format: Option<String>,
}

#[napi(object)]
pub struct JsTensorStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub shape: Vec<u32>,
    pub dtype: String,
    pub element_count: u32,
}

#[napi(object)]
pub struct JsDiffResult {
    /// Type of difference
    pub diff_type: String,

    /// Path to the changed element
    pub path: String,

    /// Old value (for Modified/TypeChanged)
    pub old_value: Option<serde_json::Value>,

    /// New value (for Modified/TypeChanged/Added)
    pub new_value: Option<serde_json::Value>,

    /// Value (for Removed)
    pub value: Option<serde_json::Value>,

    // AI/ML specific fields
    /// Old shape (for TensorShapeChanged)
    pub old_shape: Option<Vec<u32>>,

    /// New shape (for TensorShapeChanged)
    pub new_shape: Option<Vec<u32>>,

    /// Old statistics (for TensorStatsChanged)
    pub old_stats: Option<JsTensorStats>,

    /// New statistics (for TensorStatsChanged)
    pub new_stats: Option<JsTensorStats>,

    /// Old mean (for TensorDataChanged)
    pub old_mean: Option<f64>,

    /// New mean (for TensorDataChanged)
    pub new_mean: Option<f64>,

    /// Change magnitude (for WeightSignificantChange)
    pub change_magnitude: Option<f64>,

    /// Old string value (for architecture/activation/optimizer changes)
    pub old_string: Option<String>,

    /// New string value (for architecture/activation/optimizer changes)
    pub new_string: Option<String>,

    /// Old float value (for learning rate/loss/accuracy changes)
    pub old_float: Option<f64>,

    /// New float value (for learning rate/loss/accuracy changes)
    pub new_float: Option<f64>,
}

/// Unified diff function for JavaScript/Node.js
///
/// Compare two JavaScript objects or values and return differences.
///
/// # Arguments
///
/// * `old` - The old value (JavaScript object, array, or primitive)
/// * `new` - The new value (JavaScript object, array, or primitive)
/// * `options` - Optional configuration object
///
/// # Returns
///
/// Array of difference objects
///
/// # Example
///
/// ```javascript
/// const { diff } = require('diffai-js');
///
/// const oldModel = { layers: [{ weight: [1, 2, 3] }] };
/// const newModel = { layers: [{ weight: [1, 2, 4] }] };
/// const result = diff(oldModel, newModel);
/// console.log(result);
/// ```
#[napi]
pub fn diff(
    old: serde_json::Value,
    #[napi(ts_arg_type = "any")] new_value: serde_json::Value,
    options: Option<JsDiffOptions>,
) -> Result<Vec<JsDiffResult>> {
    let rust_options = options.map(build_diff_options).transpose()?;

    let results = core_diff(&old, &new_value, rust_options.as_ref())
        .map_err(|e| Error::new(Status::GenericFailure, format!("Diff error: {e}")))?;

    let js_results = results
        .into_iter()
        .map(convert_diff_result)
        .collect::<Result<Vec<_>>>()?;

    Ok(js_results)
}

/// Compare two files or directories
///
/// # Arguments
///
/// * `old_path` - Path to the old file or directory
/// * `new_path` - Path to the new file or directory
/// * `options` - Optional configuration object
///
/// # Returns
///
/// Array of difference objects
#[napi]
pub fn diff_paths(
    old_path: String,
    new_path: String,
    options: Option<JsDiffOptions>,
) -> Result<Vec<JsDiffResult>> {
    let rust_options = options.map(build_diff_options).transpose()?;

    let results = core_diff_paths(&old_path, &new_path, rust_options.as_ref())
        .map_err(|e| Error::new(Status::GenericFailure, format!("Diff error: {e}")))?;

    let js_results = results
        .into_iter()
        .map(convert_diff_result)
        .collect::<Result<Vec<_>>>()?;

    Ok(js_results)
}

/// Format diff results as string
///
/// # Arguments
///
/// * `results` - Array of diff results
/// * `format` - Output format ("diffai", "json", "yaml")
///
/// # Returns
///
/// Formatted string output
#[napi]
pub fn format_output(results: Vec<JsDiffResult>, format: String) -> Result<String> {
    let rust_results = results
        .into_iter()
        .map(convert_js_diff_result)
        .collect::<Result<Vec<_>>>()?;

    let output_format = OutputFormat::parse_format(&format)
        .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid format: {e}")))?;

    core_format_output(&rust_results, output_format)
        .map_err(|e| Error::new(Status::GenericFailure, format!("Format error: {e}")))
}

// Helper functions

fn build_diff_options(js_options: JsDiffOptions) -> Result<DiffOptions> {
    let mut options = DiffOptions::default();

    if let Some(epsilon) = js_options.epsilon {
        options.epsilon = Some(epsilon);
    }

    if let Some(array_id_key) = js_options.array_id_key {
        options.array_id_key = Some(array_id_key);
    }

    if let Some(ignore_keys_regex) = js_options.ignore_keys_regex {
        let regex = Regex::new(&ignore_keys_regex)
            .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid regex: {e}")))?;
        options.ignore_keys_regex = Some(regex);
    }

    if let Some(path_filter) = js_options.path_filter {
        options.path_filter = Some(path_filter);
    }

    if let Some(output_format) = js_options.output_format {
        let format = OutputFormat::parse_format(&output_format)
            .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid output format: {e}")))?;
        options.output_format = Some(format);
    }

    Ok(options)
}

fn convert_tensor_stats(stats: &TensorStats) -> JsTensorStats {
    JsTensorStats {
        mean: stats.mean,
        std: stats.std,
        min: stats.min,
        max: stats.max,
        shape: stats.shape.iter().map(|&s| s as u32).collect(),
        dtype: stats.dtype.clone(),
        element_count: stats.element_count as u32,
    }
}

fn convert_diff_result(result: DiffResult) -> Result<JsDiffResult> {
    match result {
        DiffResult::Added(path, value) => Ok(JsDiffResult {
            diff_type: "Added".to_string(),
            path,
            old_value: None,
            new_value: Some(value),
            value: None,
            old_shape: None,
            new_shape: None,
            old_stats: None,
            new_stats: None,
            old_mean: None,
            new_mean: None,
            change_magnitude: None,
            old_string: None,
            new_string: None,
            old_float: None,
            new_float: None,
        }),
        DiffResult::Removed(path, value) => Ok(JsDiffResult {
            diff_type: "Removed".to_string(),
            path,
            old_value: None,
            new_value: None,
            value: Some(value),
            old_shape: None,
            new_shape: None,
            old_stats: None,
            new_stats: None,
            old_mean: None,
            new_mean: None,
            change_magnitude: None,
            old_string: None,
            new_string: None,
            old_float: None,
            new_float: None,
        }),
        DiffResult::Modified(path, old_val, new_val) => Ok(JsDiffResult {
            diff_type: "Modified".to_string(),
            path,
            old_value: Some(old_val),
            new_value: Some(new_val),
            value: None,
            old_shape: None,
            new_shape: None,
            old_stats: None,
            new_stats: None,
            old_mean: None,
            new_mean: None,
            change_magnitude: None,
            old_string: None,
            new_string: None,
            old_float: None,
            new_float: None,
        }),
        DiffResult::TypeChanged(path, old_val, new_val) => Ok(JsDiffResult {
            diff_type: "TypeChanged".to_string(),
            path,
            old_value: Some(old_val),
            new_value: Some(new_val),
            value: None,
            old_shape: None,
            new_shape: None,
            old_stats: None,
            new_stats: None,
            old_mean: None,
            new_mean: None,
            change_magnitude: None,
            old_string: None,
            new_string: None,
            old_float: None,
            new_float: None,
        }),
        DiffResult::TensorShapeChanged(path, old_shape, new_shape) => Ok(JsDiffResult {
            diff_type: "TensorShapeChanged".to_string(),
            path,
            old_value: None,
            new_value: None,
            value: None,
            old_shape: Some(old_shape.iter().map(|&s| s as u32).collect()),
            new_shape: Some(new_shape.iter().map(|&s| s as u32).collect()),
            old_stats: None,
            new_stats: None,
            old_mean: None,
            new_mean: None,
            change_magnitude: None,
            old_string: None,
            new_string: None,
            old_float: None,
            new_float: None,
        }),
        DiffResult::TensorStatsChanged(path, old_stats, new_stats) => Ok(JsDiffResult {
            diff_type: "TensorStatsChanged".to_string(),
            path,
            old_value: None,
            new_value: None,
            value: None,
            old_shape: None,
            new_shape: None,
            old_stats: Some(convert_tensor_stats(&old_stats)),
            new_stats: Some(convert_tensor_stats(&new_stats)),
            old_mean: None,
            new_mean: None,
            change_magnitude: None,
            old_string: None,
            new_string: None,
            old_float: None,
            new_float: None,
        }),
        DiffResult::TensorDataChanged(path, old_mean, new_mean) => Ok(JsDiffResult {
            diff_type: "TensorDataChanged".to_string(),
            path,
            old_value: None,
            new_value: None,
            value: None,
            old_shape: None,
            new_shape: None,
            old_stats: None,
            new_stats: None,
            old_mean: Some(old_mean),
            new_mean: Some(new_mean),
            change_magnitude: None,
            old_string: None,
            new_string: None,
            old_float: None,
            new_float: None,
        }),
        DiffResult::ModelArchitectureChanged(path, old_arch, new_arch) => Ok(JsDiffResult {
            diff_type: "ModelArchitectureChanged".to_string(),
            path,
            old_value: None,
            new_value: None,
            value: None,
            old_shape: None,
            new_shape: None,
            old_stats: None,
            new_stats: None,
            old_mean: None,
            new_mean: None,
            change_magnitude: None,
            old_string: Some(old_arch),
            new_string: Some(new_arch),
            old_float: None,
            new_float: None,
        }),
        DiffResult::WeightSignificantChange(path, magnitude) => Ok(JsDiffResult {
            diff_type: "WeightSignificantChange".to_string(),
            path,
            old_value: None,
            new_value: None,
            value: None,
            old_shape: None,
            new_shape: None,
            old_stats: None,
            new_stats: None,
            old_mean: None,
            new_mean: None,
            change_magnitude: Some(magnitude),
            old_string: None,
            new_string: None,
            old_float: None,
            new_float: None,
        }),
        DiffResult::ActivationFunctionChanged(path, old_fn, new_fn) => Ok(JsDiffResult {
            diff_type: "ActivationFunctionChanged".to_string(),
            path,
            old_value: None,
            new_value: None,
            value: None,
            old_shape: None,
            new_shape: None,
            old_stats: None,
            new_stats: None,
            old_mean: None,
            new_mean: None,
            change_magnitude: None,
            old_string: Some(old_fn),
            new_string: Some(new_fn),
            old_float: None,
            new_float: None,
        }),
        DiffResult::LearningRateChanged(path, old_lr, new_lr) => Ok(JsDiffResult {
            diff_type: "LearningRateChanged".to_string(),
            path,
            old_value: None,
            new_value: None,
            value: None,
            old_shape: None,
            new_shape: None,
            old_stats: None,
            new_stats: None,
            old_mean: None,
            new_mean: None,
            change_magnitude: None,
            old_string: None,
            new_string: None,
            old_float: Some(old_lr),
            new_float: Some(new_lr),
        }),
        DiffResult::OptimizerChanged(path, old_opt, new_opt) => Ok(JsDiffResult {
            diff_type: "OptimizerChanged".to_string(),
            path,
            old_value: None,
            new_value: None,
            value: None,
            old_shape: None,
            new_shape: None,
            old_stats: None,
            new_stats: None,
            old_mean: None,
            new_mean: None,
            change_magnitude: None,
            old_string: Some(old_opt),
            new_string: Some(new_opt),
            old_float: None,
            new_float: None,
        }),
        DiffResult::LossChange(path, old_loss, new_loss) => Ok(JsDiffResult {
            diff_type: "LossChange".to_string(),
            path,
            old_value: None,
            new_value: None,
            value: None,
            old_shape: None,
            new_shape: None,
            old_stats: None,
            new_stats: None,
            old_mean: None,
            new_mean: None,
            change_magnitude: None,
            old_string: None,
            new_string: None,
            old_float: Some(old_loss),
            new_float: Some(new_loss),
        }),
        DiffResult::AccuracyChange(path, old_acc, new_acc) => Ok(JsDiffResult {
            diff_type: "AccuracyChange".to_string(),
            path,
            old_value: None,
            new_value: None,
            value: None,
            old_shape: None,
            new_shape: None,
            old_stats: None,
            new_stats: None,
            old_mean: None,
            new_mean: None,
            change_magnitude: None,
            old_string: None,
            new_string: None,
            old_float: Some(old_acc),
            new_float: Some(new_acc),
        }),
        DiffResult::ModelVersionChanged(path, old_ver, new_ver) => Ok(JsDiffResult {
            diff_type: "ModelVersionChanged".to_string(),
            path,
            old_value: None,
            new_value: None,
            value: None,
            old_shape: None,
            new_shape: None,
            old_stats: None,
            new_stats: None,
            old_mean: None,
            new_mean: None,
            change_magnitude: None,
            old_string: Some(old_ver),
            new_string: Some(new_ver),
            old_float: None,
            new_float: None,
        }),
    }
}

fn convert_js_diff_result(js_result: JsDiffResult) -> Result<DiffResult> {
    match js_result.diff_type.as_str() {
        "Added" => {
            let value = js_result.new_value.ok_or_else(|| {
                Error::new(Status::InvalidArg, "Added result must have new_value")
            })?;
            Ok(DiffResult::Added(js_result.path, value))
        }
        "Removed" => {
            let value = js_result
                .value
                .ok_or_else(|| Error::new(Status::InvalidArg, "Removed result must have value"))?;
            Ok(DiffResult::Removed(js_result.path, value))
        }
        "Modified" => {
            let old_value = js_result.old_value.ok_or_else(|| {
                Error::new(Status::InvalidArg, "Modified result must have old_value")
            })?;
            let new_value = js_result.new_value.ok_or_else(|| {
                Error::new(Status::InvalidArg, "Modified result must have new_value")
            })?;
            Ok(DiffResult::Modified(js_result.path, old_value, new_value))
        }
        "TypeChanged" => {
            let old_value = js_result.old_value.ok_or_else(|| {
                Error::new(Status::InvalidArg, "TypeChanged result must have old_value")
            })?;
            let new_value = js_result.new_value.ok_or_else(|| {
                Error::new(Status::InvalidArg, "TypeChanged result must have new_value")
            })?;
            Ok(DiffResult::TypeChanged(
                js_result.path,
                old_value,
                new_value,
            ))
        }
        "TensorShapeChanged" => {
            let old_shape = js_result.old_shape.ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "TensorShapeChanged result must have old_shape",
                )
            })?;
            let new_shape = js_result.new_shape.ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "TensorShapeChanged result must have new_shape",
                )
            })?;
            Ok(DiffResult::TensorShapeChanged(
                js_result.path,
                old_shape.iter().map(|&s| s as usize).collect(),
                new_shape.iter().map(|&s| s as usize).collect(),
            ))
        }
        "TensorDataChanged" => {
            let old_mean = js_result.old_mean.ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "TensorDataChanged result must have old_mean",
                )
            })?;
            let new_mean = js_result.new_mean.ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "TensorDataChanged result must have new_mean",
                )
            })?;
            Ok(DiffResult::TensorDataChanged(
                js_result.path,
                old_mean,
                new_mean,
            ))
        }
        "WeightSignificantChange" => {
            let magnitude = js_result.change_magnitude.ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "WeightSignificantChange result must have change_magnitude",
                )
            })?;
            Ok(DiffResult::WeightSignificantChange(
                js_result.path,
                magnitude,
            ))
        }
        "LearningRateChanged" => {
            let old_lr = js_result.old_float.ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "LearningRateChanged result must have old_float",
                )
            })?;
            let new_lr = js_result.new_float.ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "LearningRateChanged result must have new_float",
                )
            })?;
            Ok(DiffResult::LearningRateChanged(
                js_result.path,
                old_lr,
                new_lr,
            ))
        }
        "LossChange" => {
            let old_loss = js_result.old_float.ok_or_else(|| {
                Error::new(Status::InvalidArg, "LossChange result must have old_float")
            })?;
            let new_loss = js_result.new_float.ok_or_else(|| {
                Error::new(Status::InvalidArg, "LossChange result must have new_float")
            })?;
            Ok(DiffResult::LossChange(js_result.path, old_loss, new_loss))
        }
        "AccuracyChange" => {
            let old_acc = js_result.old_float.ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "AccuracyChange result must have old_float",
                )
            })?;
            let new_acc = js_result.new_float.ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "AccuracyChange result must have new_float",
                )
            })?;
            Ok(DiffResult::AccuracyChange(js_result.path, old_acc, new_acc))
        }
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("Invalid diff result type: {}", js_result.diff_type),
        )),
    }
}
