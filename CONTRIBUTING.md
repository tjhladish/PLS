
We use Github issues / pull requests workflow for contributions. We have based this guidance on the example for [docs.github.com](https://github.com/github/docs/blob/HEAD/CONTRIBUTING.md).

In this guide you will get an overview of the contribution workflow from opening an issue, creating a PR, reviewing, and merging the PR.

Use the table of contents icon on the top left corner of this document to get to a specific section of this guide quickly.

## New contributor guide

To get an overview of the project, read the [README](README.md). Here are some resources to help you get started with open source contributions:

- [Finding ways to contribute to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github)
- [Set up Git](https://docs.github.com/en/get-started/quickstart/set-up-git)
- [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Collaborating with pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests)


## Getting started

The PLS library has three elements:
 - the PLS header (pls.h), defining types used when interacting with the library and the exported interface
 - the PLS implementation (pls.cpp), which implements the methods
 - a PLS executable (pls_main.cpp), which will perform PLS on some inputs

In general, we are interested in a few kinds of contributions:
 - new cross-validation methods (currently supported: leave-one-out, leave-some-out, and against new data)
 - bug fixes
 - performance improvements
 - maintenance improvements (e.g. documentation, refactoring to use Eigen exports)

### Issues

#### Create a new issue

If you would like to contribute, [search if an issue already exists addressing your idea](https://docs.github.com/en/github/searching-for-information-on-github/searching-on-github/searching-issues-and-pull-requests#search-by-the-title-body-or-comments). If a related issue doesn't exist, you can open a new issue using a relevant [issue form](https://github.com/tjhladish/PLS/issues/new/choose).

#### Solve an issue

Scan through our [existing issues](https://github.com/tjhladish/PLS/issues) to find one that interests you. You can narrow down the search using `labels` as filters. We use the following labels:

 - `question`: something for discussion
 - `bug`: any error, e.g. in the underlying math, in code behavior, ...
 - `documentation`: issues with documentation and/or proposals to improve it
 - `performance`: issues with performance (run time, memory) and/or proposals to improve it
 - `maintenance`: issues with maintainability and/or proposals to improve it
 - `capability`: issues with PLS-adjacent functionality and/or proposals to improve it
 - `enhancement`: issues concerning some other kind of enhancement
 - for closed-without-action: `duplicate`, `invalid`, `wontfix`. Respectively: already a covered in another issue (pertinent extra content should be relocated to open issue); not actually an issue; and describes something interesting but that we won't address

As a general rule, we don’t assign issues to anyone. If you find an issue to work on, you are welcome to open a PR with a fix.

### Make Changes

#### Make changes in the UI

Click **Make a contribution** at the bottom of any docs page to make small changes such as a typo, sentence fix, or a broken link. This takes you to the `.md` file where you can make your changes and [create a pull request](#pull-request) for a review.

 <img src="/contributing/images/contribution_cta.png" />

#### Make changes in a codespace

For more information about using a codespace for working on GitHub documentation, see "[Working in a codespace](https://github.com/github/docs/blob/main/contributing/codespace.md)."

#### Make changes locally

1. Fork the repository.
- Using GitHub Desktop:
  - [Getting started with GitHub Desktop](https://docs.github.com/en/desktop/installing-and-configuring-github-desktop/getting-started-with-github-desktop) will guide you through setting up Desktop.
  - Once Desktop is set up, you can use it to [fork the repo](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/cloning-and-forking-repositories-from-github-desktop)!

- Using the command line:
  - [Fork the repo](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo#fork-an-example-repository) so that you can make your changes without affecting the original project until you're ready to merge them.

2. Follow the installation / setup guidelines (...once we have them).

3. Create a working branch and start with your changes!

4. Commit the changes once you are happy with them AND have confirmed you meet the new code requirements (...once we have them).

### Pull Request

When you're finished with the changes, create a pull request, also known as a PR.
- Create a PR. Please select the appropriate template (...once available). These templates help reviewers understand your changes as well as the purpose of your pull request.
- Don't forget to [link PR to issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) if you are solving one.
- Enable the checkbox to [allow maintainer edits](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork) so the branch can be updated for a merge.

Once you submit your PR, we will review your proposal. We may ask questions or request additional information.
- We may ask for changes to be made before a PR can be merged, either using [suggested changes](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-feedback-in-your-pull-request) or pull request comments. You can apply suggested changes directly through the web UI. You can also make any other changes locally or via the UI in your fork, then commit them to your branch.
- As you update your PR and apply changes, mark each conversation as [resolved](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request#resolving-conversations).
- If you run into any merge issues, checkout this [git tutorial](https://github.com/skills/resolve-merge-conflicts) to help you resolve merge conflicts and other issues.

### Your PR is merged!

Once your PR is merged, your contributions will be publicly attributed via the Github interface.
