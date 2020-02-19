"""Store credientials and private configs here. Add to gitignore"""

# GitHub PERSONAL ACCESS TOKEN
PAT = "<PERSONAL ACCESS TOKEN>"

# Azure SQL database config
SERVER = "mydb.database.windows.net"
DATABASE = "my_db"
USERNAME = "username"
PASSWORD = "pass"

# INSERT SQL pyodbc query. When using LEFT, NVARCHARS are interpreted as ntext.
# Must cast to NVARCHAR for truncation
INSERT_QUERY = """\
INSERT INTO my_table_v1
(
nameWithOwner,
stargazers,
createdAt,
updatedAt,
openissues,
closedissues,
forkCount,
pullRequests,
commitnum,
watchers,
diskUsage_kb,
readme_bytes,
releases,
projects,
milestones,
deployments,
primaryLanguage,
issuelabels,
repositoryTopics,
licenseInfo,
homepageUrl,
description,
hasIssuesEnabled,
hasWikiEnabled,
isLocked,
isDisabled,
id,
databaseId,
createdAt_sec,
updatedAt_sec
)
VALUES
(
LEFT(CAST(? AS NVARCHAR(200)), 200),
?,
?,
?,
?,
?,
?,
?,
?,
?,
?,
?,
?,
?,
?,
?,
LEFT(CAST(? AS NVARCHAR(200)), 200),
?,
?,
LEFT(CAST(? AS NVARCHAR(2000)), 2000),
LEFT(CAST(? AS NVARCHAR(1000)), 1000),
LEFT(CAST(? AS NVARCHAR(2000)), 2000),
?,
?,
?,
?,
?,
?,
?,
?
);
"""
