import path from 'path';
import fs from 'fs';
import { compileSync } from '@mdx-js/mdx'
import mdx from '@mdx-js/mdx';


function removeItemOnce(arr, value) {
    var index = arr.indexOf(value);
    if (index > -1) {
        arr.splice(index, 1);
    }
    return arr;
    // thanks https://stackoverflow.com/a/5767357/13090245
}

export async function articlesWithMetadata() {
    const articles_dir = path.join(process.cwd(), "./src/pages/articles");
    let articles = fs.readdirSync(articles_dir);

    articles.map((article) => {
        compileSync(article)
    })

    console.log(articles)
    // var metadata = articles.map((m) => (m.meta ? m.meta : null));
    articles.sort(
        (a, b) => new Date(a.date) - new Date(b.date)
    ).reverse();

    return articles
}