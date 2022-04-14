
import click
import torch
import dnnlib
import legacy


@click.command()
@click.argument("network-pkl")
@click.argument("output-file")
def convert(network_pkl, output_file):
    with dnnlib.util.open_url(network_pkl) as f:
        G_ema = legacy.load_network_pkl(f)['G_ema']
        for key, val in G_ema.state_dict().items():
            print(key, val.size())
        torch.save(G_ema.state_dict(), output_file)


if __name__ == "__main__":
    convert()
