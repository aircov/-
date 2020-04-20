create table if not exists hot_words (
    id int not null auto_increment,
    num varchar(255) not null comment "排名",
    query varchar(255) not null comment "热搜词",
    heat varchar(255) not null comment "热度值",
    url varchar(2555) not null comment "url链接",
    crawl_time varchar(255) not null comment "抓取时间",
    primary key (id),
    index (query)
)