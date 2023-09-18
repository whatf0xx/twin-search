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

    // let msg = String::from("\x1b[31mThis text is red,\x1b[0m this text is normal, \x1b[32mThis text is green.\x1b[0m");
    // println!("{msg}");
    // let vec: Vec<char> = msg.chars().collect();
    // println!("{:?}", vec);
}
