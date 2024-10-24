use polars::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;
use std::convert::TryFrom;
use serde::Deserialize;
use crate::graph_utils::{AsUsize, usize_to_t, to_string_chunked, to_float64_chunked, to_int64_chunked};
use pyo3_polars::derive::polars_expr;

fn build_itemset_network(
    transactions: &[(i64, usize, f64)],
    num_items: usize,
    kwargs: &AssociationRuleKwargs,
) -> Vec<ItemMetrics> {
    let mut item_metrics = vec![
        ItemMetrics {
            support_count: 0.0,
            associations: Vec::new(),
            pattern_id: 0,
            lift_score: 0.0,
        };
        num_items
    ];

    // Group by transaction
    let mut transaction_map: HashMap<i64, Vec<(usize, f64)>> = HashMap::new();
    for &(transaction_id, item_id, frequency) in transactions {
        transaction_map.entry(transaction_id)
            .or_default()
            .push((item_id, frequency));
    }

    // Calculate support for each item
    let total_transactions = transaction_map.len() as f64;
    let mut item_transactions: HashMap<usize, i64> = HashMap::new();

    // First pass: count supports
    for (_transaction_id, items) in &transaction_map {
        for &(item_id, freq) in items {
            *item_transactions.entry(item_id).or_insert(0) += 1;
            if kwargs.weighted {
                item_metrics[item_id].support_count += freq;
            } else {
                item_metrics[item_id].support_count += 1.0;
            }
        }
    }

    // Filter items that don't meet minimum support threshold
    let valid_items: HashSet<usize> = item_transactions.iter()
        .filter_map(|(&item_id, &count)| {
            let support = if kwargs.weighted {
                item_metrics[item_id].support_count
            } else {
                count as f64
            };
            if (support / total_transactions) >= kwargs.min_support {
                Some(item_id)
            } else {
                None
            }
        })
        .collect();

    // Reset metrics for invalid items
    for (item_id, metrics) in item_metrics.iter_mut().enumerate() {
        if !valid_items.contains(&item_id) {
            *metrics = ItemMetrics {
                support_count: 0.0,
                associations: Vec::new(),
                pattern_id: 0,
                lift_score: 0.0,
            };
        }
    }

    // Build associations and calculate confidence
    for items in transaction_map.values() {
        if items.len() > kwargs.max_itemset_size {
            continue;
        }

        for &(antecedent, freq1) in items {
            if !valid_items.contains(&antecedent) {
                continue;
            }

            for &(consequent, freq2) in items {
                if antecedent != consequent && valid_items.contains(&consequent) {
                    let confidence = if kwargs.weighted {
                        (freq1 * freq2) / item_metrics[antecedent].support_count
                    } else {
                        let antecedent_support = item_transactions[&antecedent] as f64;
                        antecedent_support / total_transactions
                    };

                    if confidence >= kwargs.min_confidence {
                        item_metrics[antecedent].associations.push((consequent, confidence));
                    }
                }
            }
        }
    }

    // Calculate lift scores
    for (item_id, metrics) in item_metrics.iter_mut().enumerate() {
        if valid_items.contains(&item_id) {
            metrics.lift_score = metrics.associations.iter()
                .map(|(_, confidence)| confidence)
                .sum::<f64>();
        }
    }

    // Identify patterns
    let mut current_pattern = 1;
    let mut visited = vec![false; num_items];

    for start_item in 0..num_items {
        if !visited[start_item] && valid_items.contains(&start_item) {
            let mut stack = vec![start_item];
            while let Some(item) = stack.pop() {
                if !visited[item] {
                    visited[item] = true;
                    item_metrics[item].pattern_id = current_pattern;

                    for &(associated_item, confidence) in &item_metrics[item].associations {
                        if !visited[associated_item] && confidence >= kwargs.min_confidence {
                            stack.push(associated_item);
                        }
                    }
                }
            }
            current_pattern += 1;
        }
    }

    item_metrics
}


#[derive(Deserialize)]
struct AssociationRuleKwargs {
    min_support: f64,
    min_confidence: f64,
    max_itemset_size: usize,
    weighted: bool,
}

#[derive(Clone)]
struct ItemMetrics {
    support_count: f64,
    associations: Vec<(usize, f64)>,
    pattern_id: usize,
    lift_score: f64,
}

// Process transactions function
fn process_transaction_data<T>(
    transaction_id: &Int64Chunked,
    item_id: &StringChunked,
    frequency: Option<&Float64Chunked>,
) -> PolarsResult<(HashMap<String, T>, T, Vec<(i64, T, f64)>)>
where
    T: TryFrom<usize> + Copy + PartialEq + AsUsize,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let default_freq = Float64Chunked::full(PlSmallStr::from("frequency"), 1.0, transaction_id.len());
    let freq = frequency.unwrap_or(&default_freq);

    let mut item_to_id = HashMap::new();
    let mut transaction_items = Vec::new();
    let mut id_counter: T = usize_to_t(0);

    let mut get_or_insert_id = |item: &str| -> T {
        *item_to_id.entry(item.to_string()).or_insert_with(|| {
            let id = id_counter;
            id_counter = usize_to_t(id_counter.as_usize() + 1);
            id
        })
    };

    transaction_id.iter()
        .zip(item_id.iter())
        .zip(freq.iter())
        .try_for_each(|((trans, item), freq)| -> PolarsResult<()> {
            if let (Some(t), Some(i), Some(f)) = (trans, item, freq) {
                let i_id = get_or_insert_id(i);
                transaction_items.push((t, i_id, f));
            }
            Ok(())
        })?;

    Ok((item_to_id, id_counter, transaction_items))
}

fn association_rule_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::from("association_rules").into(),
        DataType::Struct(vec![
            Field::new(PlSmallStr::from("item").into(), DataType::String),
            Field::new(PlSmallStr::from("support").into(), DataType::Float64),
            Field::new(PlSmallStr::from("lift_score").into(), DataType::Float64),
            Field::new(PlSmallStr::from("pattern").into(), DataType::UInt32),
            Field::new(PlSmallStr::from("consequents").into(), DataType::List(Box::new(DataType::String))),
            Field::new(PlSmallStr::from("confidence_scores").into(), DataType::List(Box::new(DataType::Float64))),
        ])
    ))
}

#[polars_expr(output_type_func=association_rule_output)]
fn graph_association_rules(inputs: &[Series], kwargs: AssociationRuleKwargs) -> PolarsResult<Series> {
    let transaction_id = to_int64_chunked(&inputs[0])?;
    let item_id = to_string_chunked(&inputs[1])?;
    let frequency = inputs.get(2).map(|s| to_float64_chunked(s)).transpose()?;

    type ItemId = u32;

    let (item_to_id, id_counter, transaction_items) =
        process_transaction_data::<ItemId>(&transaction_id, &item_id, frequency.as_ref())?;

    let num_items = id_counter.as_usize();

    let transaction_items: Vec<_> = transaction_items.into_iter()
        .map(|(tid, iid, freq)| (tid, iid.as_usize(), freq))
        .collect();

    let item_metrics = build_itemset_network(
        &transaction_items,
        num_items,
        &kwargs
    );

    let id_to_item: HashMap<_, _> = item_to_id
        .iter()
        .map(|(k, &v)| (v.as_usize(), k.clone()))
        .collect();

    let total_transactions = transaction_items.iter()
        .map(|(tid, _, _)| tid)
        .collect::<HashSet<_>>()
        .len() as f64;

    let mut items = Vec::new();
    let mut supports = Vec::new();
    let mut lift_scores = Vec::new();
    let mut patterns = Vec::new();
    let mut consequents = Vec::new();
    let mut confidence_scores = Vec::new();

    for (id, metrics) in item_metrics.iter().enumerate() {
        // Skip items that don't meet the minimum support threshold
        if metrics.support_count > 0.0 &&
            (metrics.support_count / total_transactions) >= kwargs.min_support {
            if let Some(item_name) = id_to_item.get(&id) {
                let mut associations = metrics.associations.clone();
                associations.sort_by(|(_, conf1), (_, conf2)| conf2.partial_cmp(conf1).unwrap());

                let (consequent_items, confidences): (Vec<_>, Vec<_>) = associations
                    .into_iter()
                    .take(5)
                    .map(|(iid, conf)| (
                        id_to_item.get(&iid).unwrap().clone(),
                        conf
                    ))
                    .unzip();

                items.push(item_name.clone());
                supports.push(metrics.support_count);
                lift_scores.push(metrics.lift_score);
                patterns.push(metrics.pattern_id as u32);
                consequents.push(Series::new(PlSmallStr::from("").into(), consequent_items));
                confidence_scores.push(Series::new(PlSmallStr::from("").into(), confidences));
            }
        }
    }

    let fields = vec![
        Series::new(PlSmallStr::from("item").into(), items),
        Series::new(PlSmallStr::from("support").into(), supports),
        Series::new(PlSmallStr::from("lift_score").into(), lift_scores),
        Series::new(PlSmallStr::from("pattern").into(), patterns),
        Series::new(PlSmallStr::from("consequents").into(), consequents),
        Series::new(PlSmallStr::from("confidence_scores").into(), confidence_scores),
    ];

    StructChunked::from_series(PlSmallStr::from("association_rules").into(), &fields)
        .map(|ca| ca.into_series())
}

