use crate::{
    data::{RandBatch, RandBatcher, RandDataset},
    model::{Model, ModelConfig},
};
use burn::data::dataloader::DataLoaderBuilder;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::record::FullPrecisionSettings;
use burn::record::JsonGzFileRecorder;
use burn::record::NamedMpkFileRecorder;
use burn::record::PrettyJsonFileRecorder;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{
    LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep, metric::LossMetric,
};
//use burn::train::metric::AccuracyMetric;

/// Quadratic loss (per sample): sum((input - target)^2, dim=1)
pub fn quadratic_loss<B: Backend>(
    specificaion_vector: Tensor<B, 1>,
    results: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let target = specificaion_vector.unsqueeze::<2>();
    let diff = target - results;
    let squared = diff.clone() * diff; // same as diff.powf(2.0)
    let summed = squared.mean_dim(1); // sum over features → shape [batch]
    summed.squeeze::<1>(1)
}

impl<B: Backend> Model<B> {
    pub fn forward_regression(
        &self,
        inputs: Tensor<B, 2>,
        specification_matrix: Tensor<B, 2>,
        specificaion_vector: Tensor<B, 1>,
    ) -> RegressionOutput<B> {
        let output = self.forward(inputs.clone());
        let results = output.matmul(specification_matrix.transpose());
        //let loss = CrossEntropyLossConfig::new()
        //.init(&output.device())
        //.forward(output.clone(), targets.clone());
        let loss = quadratic_loss(specificaion_vector, results.clone());
        RegressionOutput::new(loss, inputs, results)
    }
}

impl<B: AutodiffBackend> TrainStep<RandBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: RandBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(
            batch.inputs,
            batch.specification_matrix,
            batch.specification_vector,
        );

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<RandBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: RandBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(
            batch.inputs,
            batch.specification_matrix,
            batch.specification_vector,
        )
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 100)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = RandBatcher::<B>::new(device.clone());
    let batcher_valid = RandBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(RandDataset::generate(device.clone(), 1000));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(RandDataset::generate(device.clone(), 100));

    let learner = LearnerBuilder::new(artifact_dir)
        //.metric_train_numeric(AccuracyMetric::new())
        //.metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);
    let model_trained2 = model_trained.clone();

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
    // Save model in MessagePack format with full precision
    let recorder = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
    model_trained2
        .save_file("./models/prettymodel.json", &recorder)
        .expect("Should be able to save the model");
}
