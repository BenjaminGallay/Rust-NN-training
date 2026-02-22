use crate::{
    data::{RandBatcher, RandDataset},
    training::TrainingConfig,
};
use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
    record::{CompactRecorder, Recorder},
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, taille: usize) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let _model = config.model.init::<B>(&device).load_record(record);

    let batcher: RandBatcher<B> = RandBatcher::new(device.clone());
    let _batch = batcher.batch(
        RandDataset::generate(device.clone(), taille).vector,
        &device,
    );
    //let output = model.forward(batch.inputs);

    //println!("found {} Expected", output);
}
