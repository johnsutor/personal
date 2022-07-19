import matter from "gray-matter";
import ReactMarkdown from "react-markdown";
import Layout from "../../components/Layout";
import rehypeRaw from 'rehype-raw'
import Gist from "react-gist";


function Article({ content, frontmatter }) {

  return (
    <Layout>
      <article className="max-w-prose mx-auto">
        <header>
          <h1 className="font-extrabold text-2xl lg:text-4xl mt-8">{frontmatter.title}</h1>
          <h3 className="italic text-lg lg:text-2xl font-semibold my-2">{frontmatter.description}</h3>
        </header>
        <ReactMarkdown 
          rehypePlugins={[rehypeRaw]} 
          children={content} 
          className="prose lg:prose-lg prose-blue max-w-none mx-auto hover:prose-img:scale-150 hover:prose-img:shadow-xl hover:prose-img:z-50"
          components={{
            gist({node, inline, className, children, ...props}) {
              return (
                <Gist
                  id ={props.id}
                  {...props}
                />
              )
            }
          }}/>
      </article>
    </Layout>
  );
};

export async function getStaticPaths() {
  const fs = require('fs');
  const fileNames = fs.readdirSync("./content/articles/")
  let article_slugs =  fileNames.map(fileName => {
    return {
      params: {
        slug: fileName.replace(/\.md$/, '')
      }
    }
  })
  console.log(article_slugs)
  return { paths: article_slugs, fallback: false }
}

export async function getStaticProps(context) {
  const content = await import(`../../content/articles/${context.params.slug}.md`);
  const data = matter(content.default);
  return { props: {frontmatter: data.data, content: data.content } };
}

// Article.getInitialProps = async (context) => {
//     const content = await import(`../../content/articles/${context.query.slug}.md`);
//     const data = matter(content.default);
//     return { ...data };
// };

export default Article