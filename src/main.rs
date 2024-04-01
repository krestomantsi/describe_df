use clap::{arg, command, Parser};
use csv::Writer;
use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use polars::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(author,version,about,long_about=None)]
struct Args {
    /// describe file path
    #[arg(short, long, default_value = "result.csv")]
    file_name: PathBuf,
}

fn main() {
    let args = Args::parse();
    // ---------------  Inputs -----------------------------
    let fname = args
        .file_name
        .to_str()
        .expect("Could not parse the file path!");

    let file =
        File::open(fname).expect(format!("Could not open the specified file {:?}", fname).as_str());

    let dump = "describe_".to_string();
    let result_file = Path::new("describe_result.csv");
    println!("Writing on {:?}", &result_file);

    let mut df = CsvReader::new(file)
        .infer_schema(None)
        .has_header(true)
        .finish()
        .expect("Could not parse the specified csv!");

    let names = df.get_column_names();
    // println!("{:?}", &df);
    let result = describe_df(&df).expect("Description failed");
    println!("{:#?}", &result);

    let mut wtr = Writer::from_path(result_file).unwrap();

    for x in result.iter() {
        wtr.serialize(x).expect("Serialization failed");
    }
    wtr.flush().expect("Could not flush");
}

/// a function that returns the bin locations as well as the counts for each bin
/// to be used for most freq value calculation
pub fn histogram(x: &Vec<f32>) -> (Vec<f32>, Vec<usize>, f32) {
    let n = x.len();
    let xmin = x.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let xmax = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let nbins = (1.0 + (n as f32).log2()).ceil() as usize;
    // println!("nbins: {}", nbins);
    let xt = Array1::linspace(xmin, xmax, nbins).to_vec();

    let counts: Vec<usize> = (0..(nbins - 1))
        .into_par_iter()
        .map(|ii| count_in(xt[ii], xt[ii + 1], x))
        .collect();
    let ind = argsort(&counts);
    let ind = ind.last().expect("Could not take the last element!");
    let binding2 = n - 2;
    let ind = ind.min(&binding2);
    let most_freq_val = 0.5 * (xt[*ind] + xt[ind + 1]);
    // let most_freq_val = xt[*ind];
    (xt, counts, most_freq_val)
}

/// utility function for counting how many data are a<=x<=b
pub fn count_in(a: f32, b: f32, xdata: &Vec<f32>) -> usize {
    let mut count = 0;
    for i in xdata {
        if *i >= a && *i < b {
            count += 1;
        }
    }
    count
}

#[inline]
pub fn argsort<T: PartialOrd>(data: &[T]) -> Vec<usize> {
    // Create a vector of indices from 0 to data.len() - 1
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    // Sort the indices by comparing the corresponding elements in data using partial_cmp and unwrap_or
    indices.sort_unstable_by(|&i, &j| data[i].partial_cmp(&data[j]).unwrap_or(Ordering::Equal));
    // Return the sorted indices
    indices
}

#[test]
fn main_test() {
    //let x: Vec<f32> = (0..1000).map(|_x| random::<f32>()).collect();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let x: Vec<f32> = (0..1000)
        .map(|_x| normal.sample(&mut rand::thread_rng()))
        .collect();

    let (xbins, counts, most_freq_val) = histogram(&x);
    let mut counts = counts.into_iter().map(|x| x as u32).collect::<Vec<u32>>();
    let mut counts0 = vec![0];
    counts0.append(&mut counts);
    let mut df = DataFrame::new(vec![
        Series::new("xbins", &xbins),
        Series::new("counts", &counts0),
    ])
    .expect("Could not create the dataframe itself");
    let mut outputfile = File::create("result.csv").expect("could not create file");
    CsvWriter::new(&mut outputfile)
        .include_header(true)
        .finish(&mut df)
        .expect("Could not write Dataframe into csv!");

    println!("xbins: {:?}", &xbins);
    println!("counts: {:?}", &counts0);
    println!("most_freq_val: {:?}", &most_freq_val);
}

/// a function that iterates every column and if it is f32/f64 it calculates its statistics
fn describe_df(df: &DataFrame) -> Option<Vec<DescribeResult>> {
    let names = df.get_columns();
    let mut dumpvec: Vec<DescribeResult> = Vec::with_capacity(names.len());

    for col in df.iter() {
        let col_name = col.name();
        let dtype = col.dtype();
        if (dtype == &DataType::Float64) | (dtype == &DataType::Float32) {
            let null_count = col.null_count();
            let min = col.min().unwrap().unwrap();
            let max = col.max().unwrap().unwrap();
            let mean = col.mean().unwrap() as f32;
            let std = col.std(1).unwrap() as f32;
            let values = match dtype {
                &DataType::Float64 => df2vec64(&df, col_name),
                &DataType::Float32 => df2vec32(&df, col_name),
                _ => panic!("dtyped pattern failed"),
            };

            let (xt, counts, most_freq_val) = histogram(&values);
            dumpvec.push(DescribeResult {
                name: col_name.to_owned(),
                null_count,
                min,
                max,
                mean,
                std,
                most_freq_val,
            })
        }
    }
    Some(dumpvec)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DescribeResult {
    name: String,
    null_count: usize,
    min: f32,
    max: f32,
    mean: f32,
    std: f32,
    most_freq_val: f32,
}

#[inline]
fn df2vec64(df: &DataFrame, col: &str) -> Vec<f32> {
    df.column(col)
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .flatten()
        .map(|x| x as f32)
        .collect()
}
#[inline]
fn df2vec32(df: &DataFrame, col: &str) -> Vec<f32> {
    df.column(col)
        .unwrap()
        .f32()
        .unwrap()
        .into_iter()
        .flatten()
        .collect()
}
