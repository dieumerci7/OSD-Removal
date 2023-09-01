import click

@click.command(help="")
@click.option("--cfg", type=str, help="config file path")
def main():
    pass

if __name__ == '__main__':
    main()