import matter from "gray-matter";
import ReactMarkdown from "react-markdown";
import Layout from "../../components/Layout";

export default function Article({ content, data }) {
  const frontmatter = data;

  return (
    <Layout>
        <h1 className="font-bold text-2xl mt-8">{frontmatter.title}</h1>
        <h3 className="italic text-xl font-semibold my-2">{frontmatter.description}</h3>
        <div className="prose sm:prose-sm lg:prose-lg prose-blue max-w-none">
            <ReactMarkdown children={content} className="overflow-x-scroll"/>
        </div>
    </Layout>
  );
};


Article.getInitialProps = async (context) => {
    const content = await import(`../../content/articles/${context.query.slug}.md`);
    const data = matter(content.default);
    return { ...data };
};