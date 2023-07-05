import yake
from yake.highlight import TextHighlighter
from os.path import dirname, join, basename
from argparse import ArgumentParser


def extract(input_text,
            language="en",
            max_ngram_size=3,
            deduplication_thresold=0.3,
            deduplication_algo='seqm',
            windowSize=1,
            numOfKeywords=12,
            print_keywords=False,
            ):
    custom_kw_extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedupLim=deduplication_thresold,
        dedupFunc=deduplication_algo,
        windowsSize=windowSize,
        top=numOfKeywords,
        features=None)

    keywords = custom_kw_extractor.extract_keywords(input_text)

    if print_keywords:
        print("The keywords generated are: \n")
        for kw in keywords:
            print(kw + '\n')

    th = TextHighlighter(max_ngram_size = 3)
    input_highlighted = th.highlight(input_text, keywords)

    return keywords, input_highlighted


def keys_to_sentence(keywords, highligted_text):
    all_words = highligted_text.split(' ')
    keywords_list = []
    include = False
    i = 0
    while i < len(all_words):
        if '<kw>' in all_words[i] and not '</kw>' in all_words[i]:
            include = True
            keywords_list.append(all_words[i].split('<kw>')[-1])
        if '<kw>' not in all_words[i] and '</kw>' in all_words[i]:
            include = False
            keywords_list.append(all_words[i].split('</kw>')[0])
        if '<kw>' in all_words[i] and '</kw>' in all_words[i]:
            print()
            keywords_list.append(all_words[i].replace('<kw>', '').replace('</kw>', '').replace('"', ''))
        if '<kw>' not in all_words[i] and '</kw>' not in all_words[i]:
            if include:
                keywords_list.append(all_words[i])
        i += 1

    sen = ' '.join(keywords_list)
    for key in keywords:
        num_appear = sen.count(key[0])
        if num_appear > 1:
            sen = sen.replace(key[0], 'mask_by_john_do_not_touch', 1)
            sen = sen.replace(key[0], '')
            sen = sen.replace('mask_by_john_do_not_touch', key[0]).strip()

    return sen


# def yake1file(file):
#     with open(file, 'r') as fopen:
#         text = fopen.read()
#         assert len(text) > 0, "Input file is empty!!!"
#
#     keywords, text_highlighted = extract(text)
#     keysen = keys_to_sentence(keywords, text_highlighted)
#
#     with open(join(dirname(file), 'yake-keywords.txt'), 'w') as fopen:
#         for item in keywords:
#             fopen.write(item[0] + '\n')
#         fopen.close()
#
#     with open(join(dirname(file), 'yake-keywords_in_sentence.txt'), 'w') as fopen:
#         fopen.write(keysen)
#         fopen.close()


def yake1file(file):
    with open(file, 'r') as fopen:
        text = fopen.read()
        assert len(text) > 0, "Input description file is empty!!!"

    keywords, text_highlighted = extract(text)
    keysen = keys_to_sentence(keywords, text_highlighted)

    lines_to_write = [
        text,
        "Keywords extracted:" + '\t' + ";".join([item[0] for item in keywords]),
        "Keywords in sentence extracted:" + '\t' + keysen]

    with open(file, 'w') as fopen:
        for line in lines_to_write:
            fopen.write(line + '\n')
        fopen.close()

    return keywords, keysen


if __name__ == "__main__":
    parser = ArgumentParser('Yake keywords extractor')
    parser.add_argument('--path_to_file', '-p', type=str, default=None,
                        help="path to input file")
    args = parser.parse_args()
    yake1file(args.path_to_file)


