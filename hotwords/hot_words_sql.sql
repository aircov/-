create table if not exists hot_words (
    id int not null auto_increment,
    num varchar(255) not null comment "排名",
    query varchar(255) not null comment "热搜词",
    heat varchar(255) not null comment "热度值",
    url varchar(2555) not null comment "url链接",
    crawl_time varchar(255) not null comment "抓取时间",
    source varchar(255) comment "来源",
    primary key (id),
    index (query)
)


select query,heat,crawl_time,source from hot_words where heat>1012377;

-- 查询一周
select * from hot_words  where DATE_SUB(CURDATE(), INTERVAL 7 DAY) <= date(crawl_time);