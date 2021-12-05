import matter from "gray-matter";
import ReactMarkdown from "react-markdown";
import Layout from "../../components/Layout";

export default function Article({ content, data }) {
  const frontmatter = data;

  return (
    <Layout>
      <article className="max-w-prose mx-auto">
        <header>
          <h1 className="font-extrabold text-2xl lg:text-4xl mt-8">{frontmatter.title}</h1>
          <h3 className="italic text-lg lg:text-2xl font-semibold my-2">{frontmatter.description}</h3>
        </header>
        <ReactMarkdown children={content} className="prose prose-sm lg:prose-lg prose-blue max-w-none mx-auto"/>
      </article>
    </Layout>
  );
};


Article.getInitialProps = async (context) => {
    const content = await import(`../../content/articles/${context.query.slug}.md`);
    const data = matter(content.default);
    return { ...data };
};