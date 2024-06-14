from src.cnn_lstm_model import CnnLstmModel, create_cnn_lstm_model
from src.history.analyse import analyse_history
from src.training.autoencoders import build_latent_features
from src.training.backtesting import perform_backtesting
from src.training.contrastive_learning import train_contrastive_model


def run():
    # combined_features = analyse_history()
    # latent_features = build_latent_features(combined_features)
    # enhanced_features = train_contrastive_model(latent_features)
    # model: CnnLstmModel = create_cnn_lstm_model(enhanced_features, combined_features)
    # perform_backtesting(model.predictions_dataframe, enhanced_features)
    perform_backtesting(None, None, None)


if __name__ == "__main__":
    run()
