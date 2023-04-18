import Layout from "@/components/Layout";
import { useState } from "react";

export default function Press() {
  const [press, setPress] = useState([
    {
      title: "In Conversation with John Sutor and Carlos Mercado-Lara of SciTeens",
      organization: "Youth STEM 2030",
      snippet: "Recently, Youth STEM Matters Volunteers Tashinga Mutemachani and Louise Robertson had the honour of interviewing John Sutor and Carlos Mercado-Lara, co-founders of the non-profit SciTeens. The COVID-19 pandemic meant we had to speak to them via Zoom, but that didn’t stop them from sharing some very valuable insights and advice!  In the interview, John and Carlos shared what SciTeens is all about, their journey and achievements in establishing it and an insight into their future plans.",
      url: "https://www.youthstem2030.org/youth-stem-matters/read/sciteens-interview",
      image: "https://pbs.twimg.com/profile_images/1234886467123662850/H9DLCQci_400x400.jpg"
    },
    {
      title: "FSU Student Stars",
      organization: "Florida State University",
      snippet: "John Sutor was a junior in high school when he first visited Florida State University as a participant in the Young Scholars Program, a six- week residential science and mathematics summer program for Florida high school students with significant potential for careers in science, technology, engineering and mathematics(STEM).",
      url: "https://news.fsu.edu/student-stars/2021/02/03/john-sutor/",
      image: "https://news.fsu.edu/wp-content/uploads/2021/02/John-Sutor.jpg",
    },
    {
      title: "SciTeens: Data Science and Ecology for Gen Z",
      organization: "National Ecological Observatory Network",
      snippet: "SciTeens is the brainchild of founders Carlos Mercado-Lara and John Sutor, who started the organization in 2018 when they were high school seniors. Their mission is to make free STEM resources – including data science resources – accessible to all students through online curricula, outreach, and mentoring. They leveraged ecological data from the NEON program to create their first data science projects.",
      url: "https://www.neonscience.org/impact/observatory-blog/sciteens-data-science-and-ecology-gen-z",
      image: "https://www.neonscience.org/sites/default/files/styles/_edit_list_additional_actions_max_width_300/public/2021-02/sciteens%20headshot.jpg?itok=9V-XtCuU",
    },
    {
      title: "iSchool Professor Studies Artificial Intelligence",
      organization: "Florida State University CCI",
      snippet: "Dr. Jonathan Adams and his team of Computer Science students are working hard on developing applications for Artificial Intelligence (AI). “We were awarded a grant by the Student Technology Fee Advisory Committee, which allowed us to buy a computer designed for machine learning.” Dr. Adams explains. He then recruited two students from the Undergraduate Research Opportunity Program (UROP) and an intern to develop research projects using the computer to recognize objects with a video camera. “We learned a lot about data management, machine training, and transfer learning, so our research focused on describing the training process from a work-flow perspective,” Dr. Adams says. Transfer learning tries to build on training the computer has already received, requiring less data, and less time to train the machine to recognize new objects.",
      url: "https://news.cci.fsu.edu/cci-news/cci-faculty/ischool-professor-develops-field-of-artificial-intelligence/",
      image: "https://news.cci.fsu.edu/files/2020/04/John-Adams-group.png",
    },
    {
      title: "MIT IDEAS That Inspire Podcast",
      organization: "Massachusets Institute of Technology",
      snippet: "Carlos Mercado- Lara and John Sutor are MIT - IDEAS award recipients and Co - founders of SciTeens.⁠ SciTeens is an organization with a mission to bridge the gap between education and opportunity by providing students with mentorship and community on a free online platform.",
      url: "https://podcasters.spotify.com/pod/show/ideasthatmatter/episodes/Episode-07--Dustin-Liu-in-conversation-with-Carlos-Mercado-Lara-and-John-Sutor-e1021tk",
      image: "https://s3-us-west-2.amazonaws.com/anchor-generated-image-bank/production/podcast_uploaded_nologo400/12467323/12467323-1616873097229-42e623be70c86.jpg",
    },
    {
      title: "Student Spotlight: John Sutor",
      organization: "Florida State University",
      snippet: "John Sutor is a senior double-majoring in computational science through the Department of Scientific Computing and applied mathematics through the Department of Mathematics, both part of Florida State University’s College of Arts and Sciences. Currently, he and his research team are focused on artificial intelligence as a teaching tool and the use of neural networks to produce synthetic visual data. Sutor founded SciTeens, a nonprofit that provides educational mentoring to under-resourced students. He has also published three papers about machine learning and the training of a YOLOv3 program to identify real or synthetic sea turtles in the past two years. He has spoken at the three correlating international conferences in Waynesville, N.C., Okayama, Japan, and Chengdu, China.",
      url: "https://artsandsciences.fsu.edu/article/student-spotlight-john-sutor",
      image: "https://artsandsciences.fsu.edu/sites/g/files/upcbnu321/files/2021-09/John-Sutor.jpg",
    }
  ]);
  return (

    <Layout>
      <div className="w-full text-xl mt-8">
        <h2 className="text-4xl font-bold">
          Press
        </h2>
        {press.map((article, i) => (
          <a href={article.url} key={i}>
            <div className="py-2 flex flex-col lg:flex-row items-center my-4 border-b-2">
              <div className="lg:w-1/4 object-contain">
                <img src={article.image} className="w-full h-full object-contain lg:rounded-md" />
              </div>
              <div className="mt-4 lg:ml-4 lg:w-3/4">
                <h2 className="font-semibold text-2xl">{article.title}</h2>
                <p className="italic text-gray-700">{article.organization}</p>
                <p>{article.snippet}</p>
              </div>
            </div>
          </a>
        ))}
      </div>


    </Layout>
  )
}