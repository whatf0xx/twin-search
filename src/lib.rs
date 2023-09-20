use itertools::Itertools;
use std::collections::HashMap;
use std::{fs, fmt};

fn nearest_bipartite_neighbour(x: &isize, target: &Vec<isize>) -> isize {
    // Assuming target is sorted, find the insert position in the list,
    // and then take the smallest of the distances to the nearest neighbour.
    if *x <= target[0] { return target[0] };
    
    let length = target.len();
    if *x >= target[length-1] { return target[length-1] };
    
    let mut left = 0;
    let mut right = length;
    let mut guess = length / 2;

    while right - left > 1 {
        if *x == target[guess] {
            return target[guess]
        } else if *x > target[guess] {
            left = guess;
            guess = (left + right) / 2;
        } else if *x < target[guess] {
            right = guess;
            guess = (left + right) / 2;
        }
    }

    if (*x - target[right]).abs() < (*x - target[left]).abs() { target[right] }
    else { target[left] }
}

fn find_nearest_neighbours(xs: &Vec<isize>, ys: &Vec<isize>) -> Vec<(isize, isize)> {
        let unsanitized: Vec<(isize, isize)> = xs
            .iter()
            .map(|x| (*x, nearest_bipartite_neighbour(&x, &ys)))
            .sorted_unstable_by_key(|(x, y)| (*x - *y).abs())
            .collect();

        let mut right: Vec<isize> = Vec::new();
        let mut sanitized: Vec<(isize, isize)> = Vec::new();

        unsanitized
            .into_iter()
            .for_each(
                |(l, r)| {
                    if !right.contains(&r) {
                        right.push(r.to_owned()); 
                        sanitized.push((l, r));
                    }
                }
            );

        sanitized
}

fn make_occurences_table(word: &str) -> HashMap<char, Vec<usize>> {
    let mut occurences: HashMap<char, Vec<usize>> = HashMap::new();
    word.chars().enumerate().for_each(
        |(i, c)| {
            if let Some(positions) = occurences.get_mut(&c) {
                positions.push(i);
            } else {
                occurences.insert(c, vec![i]);
            }
        }
    );

    occurences
}

fn make_bad_chars_table(key: &str) -> HashMap<char, HashMap<usize, Option<usize>>> {
    let occurences = make_occurences_table(&key);
    let mut bad_chars: HashMap<char, HashMap<usize, Option<usize>>> = HashMap::new();

    let key_length = key.len();

    occurences.keys().for_each(
        |c| {
            let mut shifts: HashMap<usize, Option<usize>> = HashMap::new();
            let mut j: usize = 0;
            for i in 0..key_length {
                let positions: Vec<usize> = occurences.get(&c).unwrap().to_vec();
                if i == positions[j] {
                    shifts.insert(i, None);  // this is a 'blank' value that should never normally be found
                    j += if j < (positions.len() - 1) { 1 } else { 0 };
                } else if i < positions[j] {
                    shifts.insert(i, Some(key_length));
                } else {
                    shifts.insert(i, Some(i - positions[j]));
                }
            }

            bad_chars.insert(*c, shifts);
        }
    );

    bad_chars

}

fn find_occurences_in_text(key: &str, text: &str) -> Vec<usize> {
    let text_chars = text.chars().collect::<Vec<char>>();
    let key_chars = key.chars().collect::<Vec<char>>();

    let text_length = text_chars.len();
    let key_length = key_chars.len();
    
    let bad_chars_table = make_bad_chars_table(key);
    
    let mut indices = Vec::new();
    let mut i = key_length - 1;

    while i < text_length {
        let mut success_flag = false;
        for j in 0..key_length {  // check backwards that the characters match
            if text_chars[i - j] == key_chars[key_length - 1 - j] {// good, continue
                if j == key_length - 1 { success_flag = true; i += key_length; }  // undefined if key can i.e., overlap with itself?
                continue
            } else {  // no match, work out how much to shift and then break the inner loop
                if let Some(shift_table) = bad_chars_table.get(&text_chars[i-j]) {
                    if let Some(Some(shift)) = shift_table.get(&j) {  // this is a bit horrible...
                        i += shift;
                    }  // this loop should never fail
                } else {  // but from here we access letters in text not in key, i.e., shift by the key length
                    i += key_length;
                }
                break;
            }
        }
        if success_flag {
            indices.push(i - key_length);
        }
    }


    indices
}

fn search_text(key1: &str, key2: &str, text: &str) -> Vec<(usize, usize)>{
    let positions1: Vec<usize> = find_occurences_in_text(key1, text);
    let positions2: Vec<usize> = find_occurences_in_text(key2, text);
    
    let nums1: Vec<isize> = positions1.into_iter().map(|x| x as isize).collect();
    let nums2: Vec<isize> = positions2.into_iter().map(|x| x as isize).collect();

    let signed_result: Vec<(isize, isize)> = find_nearest_neighbours(&nums1, &nums2);
    signed_result
        .into_iter()
        .map(|(x, y)|  (x as usize, y as usize))
        .collect()
}

fn get_sentence_start(index: usize, text: &str) -> usize {
    let mut i = index.clone();
    let characters: Vec<char> = text.chars().collect();
    let mut current: char = characters[index];
    while current != '.' && i != 0 {
        i -= 1;
        current = characters[i];
    }

    i += 1;
    if characters[i] == '\'' {
        i += 1;
    }

    loop {
        match characters[i] {
            ' ' => i += 1,
            '\n' => i += 1,
            _ => break,
        }
    }

    i
}

fn get_sentence_end(index: usize, text: &str) -> usize {
    let mut i = index.clone();
    let characters: Vec<char> = text.chars().collect();
    let mut current: char = characters[index];
    while current != '.' && i != characters.len() - 1 {
        i += 1;
        current = characters[i];
    }
    
    i
}

fn get_nearby_text(i1: usize, i2: usize, text: &str) -> String{
    let (word1, word2) = if i1 <= i2 { (i1, i2) }
                        else { (i2, i1) };
    let text_chars: Vec<char> = text.chars().collect();
    let start: usize = get_sentence_start(word1, &text);
    let end: usize = get_sentence_end(word2, &text);

    text_chars[start..end+1]
        .iter().collect()
}

fn format_sentence(sentence: &str, red_word: &str, green_word: &str) -> String {
    let mut s_chars: Vec<char> = sentence.chars().collect();
    let red_length: usize = red_word.len();
    let green_length: usize = green_word.len();

    let red_starts: Vec<usize> = find_occurences_in_text(&red_word, &sentence);

    let mut offset: usize = 0;

    red_starts.iter().for_each(
        |start| {
            // x1b[31m
            s_chars.insert(*start + offset, 'm');
            s_chars.insert(*start + offset, '1');
            s_chars.insert(*start + offset, '3');
            s_chars.insert(*start + offset, '[');
            s_chars.insert(*start + offset, '\u{1b}');

            // x1b[0m
            s_chars.insert(*start + 5 + red_length + offset, 'm');
            s_chars.insert(*start + 5 + red_length + offset, '0');
            s_chars.insert(*start + 5 + red_length + offset, '[');
            s_chars.insert(*start + 5 + red_length + offset, '\u{1b}');

            offset += 9;
        }
    );

    let temp_sentence: String = s_chars.iter().collect();
    let green_starts: Vec<usize> = find_occurences_in_text(&green_word, &temp_sentence);

    let mut offset: usize = 0;

    green_starts.iter().for_each(
        |start| {
            // x1b[32m
            s_chars.insert(*start + offset, 'm');
            s_chars.insert(*start + offset, '2');
            s_chars.insert(*start + offset, '3');
            s_chars.insert(*start + offset, '[');
            s_chars.insert(*start + offset, '\u{1b}');

            // x1b[0m
            s_chars.insert(*start + 5 + green_length + offset, 'm');
            s_chars.insert(*start + 5 + green_length + offset, '0');
            s_chars.insert(*start + 5 + green_length + offset, '[');
            s_chars.insert(*start + 5 + green_length + offset, '\u{1b}');

            offset += 9;
        }
    );
    s_chars.iter().collect()
}

pub fn run(config: Config) -> Result<(), GrepError> {
    let contents = fs::read_to_string(config.file_path).map_err(GrepError::BadFilePathError)?;

    let hits = search_text(&config.key1, &config.key2, &contents);
    println!("\n-------------------------------------------\n");
    for i in 0..10 {
        let sentence = get_nearby_text(hits[i].0, hits[i].1, &contents);
        let fmt_sentence = format_sentence(&sentence, &config.key1, &config.key2);

        println!("{}", fmt_sentence);
        // println!("{:?}", fmt_sentence.chars().collect::<Vec<char>>());
        println!("\n-------------------------------------------\n");
    }

    Ok(())
}

pub struct Config {
    pub key1: String,
    pub key2: String,
    pub file_path: String,
}

impl Config {
    pub fn build(args: &[String]) -> Result<Config, GrepError> {
        // --snip--
        if args.len() < 4 {
            return Err(GrepError::BadArgsError(args.len()));
        }

        let key1 = args[1].clone();
        let key2 = args[2].clone();
        let file_path = args[3].clone();

        Ok(Config { key1, key2, file_path })
    }
}

#[derive(Debug)]
pub enum GrepError{
    BadArgsError(usize),
    BadFilePathError(std::io::Error)
}

impl fmt::Display for GrepError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GrepError::BadArgsError(num) => write!(f, "Couldn't handle {} args, expected 3", num-1),
            GrepError::BadFilePathError(inner) => write!(f, "File read error: {}", inner),
            // Add formatting for other error variants here
        }
    }
}

#[cfg(test)]
mod tests{
    use itertools::izip;
    use super::*;

    #[test]
    fn test_nearest_bipartite_neighbour(){
        let xs: Vec<isize> = vec![15, 2, -1, 1, 23];
        let targets: Vec<Vec<isize>> = vec![
                                        vec![1, 3, 4, 6, 7, 11, 13, 16, 17, 19, 21, 28],
                                        vec![0, 1, 3, 4],
                                        vec![-2, 1, 3, 4],
                                        vec![0],
                                        vec![12, 18, 43, 44]
                                            ];

        let solutions: Vec<isize> = vec![16, 1, -2, 0, 18];

        izip!(xs, targets, solutions).for_each(
            |(x, target, solution)| { assert_eq!(nearest_bipartite_neighbour(&x, &target), solution); }
        );
    }

    #[test]
    fn test_find_nearest_neighbour(){
        let xs: Vec<Vec<isize>> = vec![
                                        vec![1, 2, 5],
                                        vec![0, 4, 9],
                                            ];

        let ys: Vec<Vec<isize>> = vec![
                                        vec![-4, 3, 9],
                                        vec![-9, -8, -2, 6, 8]
                                            ];

        let zs: Vec<(isize, isize)> = vec![(2, 3), (9, 8)];

        izip!(xs, ys, zs).for_each(
            |(x, y, z)| { 
                let sol = find_nearest_neighbours(&x, &y)[0];
                let (sol1, sol2) = sol;
                assert_eq!((sol1.to_owned(), sol2.to_owned()), z);
            }
        );
    }

    #[test]
    fn test_find_in_text(){
        let text: &str = "The quick brown fox jumped over the lazy dog. The quicker cat shot away.";
        let key: &str = "quick";

        assert_eq!(find_occurences_in_text(key, text), vec![4, 50]);
    }

    #[test]
    fn test_double_search(){
        let text: &str = "The quick brown fox jumped over the lazy dog. The quicker cat shot away.";
        let key1: &str = "quick";
        let key2: &str = "fox";

        assert_eq!(search_text(key1, key2, text), vec![(4, 16)]);
    }
}