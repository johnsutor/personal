import type { Metadata, Site, Socials } from "@types";

export const SITE: Site = {
  TITLE: "John Sutor",
  DESCRIPTION: "John Sutor's personal website.",
  EMAIL: "johnsutor3@gmail.com",
  NUM_POSTS_ON_HOMEPAGE: 5,
  NUM_PROJECTS_ON_HOMEPAGE: 5,
};

export const HOME: Metadata = {
  TITLE: "John Sutor's Home Page",
  DESCRIPTION: "Home page of John Sutor's personal website.",
};

export const PRESS: Metadata = {
  TITLE: "John Sutor's Press",
  DESCRIPTION: "A collection of my press mentions, or articles about me.",
};

export const BLOG: Metadata = {
  TITLE: "John Sutor's Blog",
  DESCRIPTION: "A collection of my blog posts.",
};

export const PROJECTS: Metadata = {
  TITLE: "John Sutor's Blog Projects",
  DESCRIPTION:
    "A collection of my projects with links to repositories and live demos.",
};

export const SOCIALS: Socials = [
  {
    NAME: "GitHub",
    HREF: "https://github.com/johnsutor",
  },
  {
    NAME: "HuggingFace",
    HREF: "https://huggingface.co/johnsutor",
  },
  {
    NAME: "LinkedIn",
    HREF: "https://www.linkedin.com/in/johnsutor3",
  },
  {
    NAME: "Instagram",
    HREF: "https://instagram.com/john_sutor",
  },
];
