use std::{env, process};
use double_grep::Config;

fn main() {
    let args: Vec<String> = env::args().collect();

    let config = Config::build(&args).unwrap_or_else(|err| {
        println!("Problem parsing arguments: {err}");
        process::exit(1);
    });

    println!("Searching for '{}' close to '{}',", config.key1, config.key2);
    println!("In file '{}'", config.file_path);

    if let Err(e) = double_grep::run(config) {
        println!("Application error: {e}");
        process::exit(1);
    }
}
