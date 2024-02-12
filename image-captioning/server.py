import flwr as fl
from Model import model, processor

weights =  [val.cpu().numpy() for _, val in model.state_dict().items()]

parameters = fl.common.ndarrays_to_parameters(weights)

strategy = fl.server.strategy.FedAvg(
    initial_parameters=parameters,
)


fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    grpc_max_message_length=2000000000,
    strategy=strategy,
)

model.save_pretrained('captioning_model')
processor.save_pretrained('captioning_processor')